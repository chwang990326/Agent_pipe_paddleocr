"""LangGraph-based workflow for the industrial multi-agent inspection system."""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.brain import ControllerBrain
from agent.config import AgentConfig
from agent.memory import AgentMemory
from agent.multi_agent import (
    DispatchAgent,
    IncidentInvestigationAgent,
    PerceptionAgent,
    QualityAgent,
    RAGAgent,
)
from agent.schemas import AgentDecision, PipeInspectionState
from agent.tools.base import ToolContext


class WorkflowState(TypedDict):
    state: PipeInspectionState


class PipeInspectionWorkflow:
    """Builds and runs the LangGraph state machine for pipe inspection."""

    def __init__(
        self,
        config: AgentConfig,
        tools: dict[str, object],
        memory: AgentMemory,
        brain: ControllerBrain,
    ):
        self.config = config
        self.tools = tools
        self.memory = memory
        self.brain = brain
        self.perception_agent = PerceptionAgent(tools)
        self.quality_agent = QualityAgent(config, tools)
        self.rag_agent = RAGAgent(config, tools)
        self.dispatch_agent = DispatchAgent(tools)
        self.incident_agent = IncidentInvestigationAgent(config, tools, memory)
        self.graph = self._build_graph()

    def run(
        self,
        task: str,
        frame_path: str | None,
        workstation: str,
        component_id: str | None = None,
        batch_id: str | None = None,
    ) -> PipeInspectionState:
        state = PipeInspectionState.create(
            task=task,
            workstation=workstation,
            component_id=component_id,
            batch_id=batch_id,
            frame_path=frame_path,
        )
        result = self.graph.invoke({"state": state})
        return result["state"]

    def _build_graph(self):
        graph = StateGraph(WorkflowState)

        graph.add_node("init", self._node_init)
        graph.add_node("perception", self._node_perception)
        graph.add_node("planning", self._node_planning)
        graph.add_node("quality", self._node_quality)
        graph.add_node("duplicate_check", self._node_duplicate_check)
        graph.add_node("rag_review", self._node_rag_review)
        graph.add_node("dispatch", self._node_dispatch)
        graph.add_node("incident_investigation", self._node_incident_investigation)
        graph.add_node("persist", self._node_persist)
        graph.add_node("error", self._node_error)

        graph.add_edge(START, "init")
        graph.add_edge("init", "perception")
        graph.add_edge("perception", "planning")
        graph.add_edge("planning", "quality")
        graph.add_edge("quality", "duplicate_check")
        graph.add_conditional_edges(
            "duplicate_check",
            self._route_after_duplicate_check,
            {
                "persist": "persist",
                "rag_review": "rag_review",
                "dispatch": "dispatch",
                "error": "error",
            },
        )
        graph.add_edge("rag_review", "dispatch")
        graph.add_edge("dispatch", "incident_investigation")
        graph.add_edge("incident_investigation", "persist")
        graph.add_edge("persist", END)
        graph.add_edge("error", "persist")

        return graph.compile()

    def _node_init(self, workflow_state: WorkflowState) -> WorkflowState:
        state = workflow_state["state"]
        state.add_trace("triggered", "SupervisorAgent.received_task", {"task": state.task})
        return {"state": state}

    def _node_perception(self, workflow_state: WorkflowState) -> WorkflowState:
        state = workflow_state["state"]
        try:
            self.perception_agent.run(state, self._context(state))
            return {"state": state}
        except Exception as exc:  # pragma: no cover - defensive workflow boundary
            return {"state": self._mark_error(state, str(exc))}

    def _node_planning(self, workflow_state: WorkflowState) -> WorkflowState:
        state = workflow_state["state"]
        if self._has_error(state):
            return {"state": state}
        plan = self.brain.plan(state)
        state.add_trace(
            "reasoning",
            "SupervisorAgent.plan_created",
            {"steps": plan.steps, "reason_summary": plan.reason_summary},
        )
        return {"state": state}

    def _node_quality(self, workflow_state: WorkflowState) -> WorkflowState:
        state = workflow_state["state"]
        if self._has_error(state):
            return {"state": state}
        try:
            decision = self.quality_agent.run(state, self._context(state))
            plan = self._plan_reason_summary(state)
            decision.reason_summary = f"{plan} {decision.reason_summary}".strip()
            state.decision = decision
            return {"state": state}
        except Exception as exc:  # pragma: no cover - defensive workflow boundary
            return {"state": self._mark_error(state, str(exc))}

    def _node_duplicate_check(self, workflow_state: WorkflowState) -> WorkflowState:
        state = workflow_state["state"]
        if self._has_error(state):
            return {"state": state}

        signature = self._signature(state)
        if self.memory.is_duplicate(signature):
            state.decision = AgentDecision(
                status="duplicate",
                action="skip_duplicate_component",
                reason_summary="The current component signature already exists in short-term memory.",
            )
            state.add_trace("finished", "SupervisorAgent.duplicate_skipped", {"signature": signature})
            return {"state": state}

        state.add_trace("reasoning", "SupervisorAgent.duplicate_check_passed", {"signature": signature})
        return {"state": state}

    def _node_rag_review(self, workflow_state: WorkflowState) -> WorkflowState:
        state = workflow_state["state"]
        if self._has_error(state):
            return {"state": state}
        try:
            decision = self.rag_agent.run(state, self._context(state), state.decision)
            state.decision = decision
            return {"state": state}
        except Exception as exc:  # pragma: no cover - defensive workflow boundary
            return {"state": self._mark_error(state, str(exc))}

    def _node_dispatch(self, workflow_state: WorkflowState) -> WorkflowState:
        state = workflow_state["state"]
        if self._has_error(state):
            return {"state": state}
        try:
            self.dispatch_agent.run(state, self._context(state), state.decision)
            return {"state": state}
        except Exception as exc:  # pragma: no cover - defensive workflow boundary
            return {"state": self._mark_error(state, str(exc))}

    def _node_incident_investigation(self, workflow_state: WorkflowState) -> WorkflowState:
        state = workflow_state["state"]
        if self._has_error(state):
            return {"state": state}
        try:
            self.incident_agent.run(state, self._context(state))
            return {"state": state}
        except Exception as exc:  # pragma: no cover - defensive workflow boundary
            return {"state": self._mark_error(state, str(exc))}

    def _node_error(self, workflow_state: WorkflowState) -> WorkflowState:
        return workflow_state

    def _node_persist(self, workflow_state: WorkflowState) -> WorkflowState:
        state = workflow_state["state"]
        if state.decision and state.decision.status not in {"duplicate", "error"}:
            signature = self._signature(state)
            self.memory.remember(signature, state)
        if not state.finished_at and state.trace:
            state.finished_at = state.trace[-1].timestamp
        if state.phase not in {"finished", "error"}:
            state.add_trace("finished", "SupervisorAgent.persist_state")
        if not state.finished_at:
            state.finished_at = state.trace[-1].timestamp if state.trace else None
        self.memory.persist_state(state)
        return {"state": state}

    def _route_after_duplicate_check(
        self,
        workflow_state: WorkflowState,
    ) -> Literal["persist", "rag_review", "dispatch", "error"]:
        state = workflow_state["state"]
        if self._has_error(state):
            return "error"
        if not state.decision:
            return "error"
        if state.decision.status == "duplicate":
            return "persist"
        if state.decision.status == "matched":
            return "rag_review"
        return "dispatch"

    def _context(self, state: PipeInspectionState) -> ToolContext:
        return ToolContext(config=self.config, run_id=state.run_id, memory=self.memory)

    def _signature(self, state: PipeInspectionState) -> str:
        if state.component_id and not state.component_id.startswith("component-"):
            return f"{state.workstation}:{state.component_id}"
        shape = state.shape_result.label if state.shape_result else "unknown"
        material_id = state.material_id or ""
        return f"{state.workstation}:{material_id}:{shape}"

    @staticmethod
    def _has_error(state: PipeInspectionState) -> bool:
        return bool(state.decision and state.decision.status == "error")

    def _mark_error(self, state: PipeInspectionState, message: str) -> PipeInspectionState:
        state.decision = AgentDecision(
            status="error",
            action="suspend_for_operator",
            reason_summary=message,
            alert_required=True,
            suspend_for_human=True,
        )
        state.add_trace("error", "LangGraph.workflow_exception", {"error": message})
        return state

    @staticmethod
    def _plan_reason_summary(state: PipeInspectionState) -> str:
        for event in reversed(state.trace):
            if event.action == "SupervisorAgent.plan_created":
                value = event.observation.get("reason_summary")
                if isinstance(value, str):
                    return value
        return ""

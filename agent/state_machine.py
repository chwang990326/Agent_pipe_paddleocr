"""LangGraph-like ReAct state machine for pipe inspection."""

from __future__ import annotations

from agent.brain import ControllerBrain
from agent.config import AgentConfig
from agent.memory import AgentMemory
from agent.parsing import normalize_material_id, parse_pipe_text
from agent.schemas import (
    AgentDecision,
    ERPRecord,
    OCRResult,
    PipeInspectionState,
    ShapeResult,
    new_id,
    now_iso,
)
from agent.tools.base import ToolContext


class PipeInspectionWorkflow:
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

    def run(
        self,
        task: str,
        frame_path: str | None,
        workstation: str,
        component_id: str | None = None,
        batch_id: str | None = None,
    ) -> PipeInspectionState:
        state = PipeInspectionState(
            run_id=new_id("run"),
            task=task,
            workstation=workstation,
            component_id=component_id or new_id("component"),
            batch_id=batch_id or new_id("batch"),
            frame_path=frame_path,
        )
        context = ToolContext(config=self.config, run_id=state.run_id, memory=self.memory)

        try:
            state.add_trace("triggered", "sensor_event_received", {"task": task})
            self._perceive(state, context)
            signature = self._signature(state)
            if self.memory.is_duplicate(signature):
                state.decision = AgentDecision(
                    status="duplicate",
                    action="skip_duplicate_component",
                    reason_summary="The current component signature already exists in short-term memory.",
                )
                state.add_trace("finished", "duplicate_skipped", {"signature": signature})
                return self._finish(state)

            plan = self.brain.plan(state)
            state.add_trace(
                "reasoning",
                "brain_plan_created",
                {"steps": plan.steps, "reason_summary": plan.reason_summary},
            )

            self._validate_with_erp(state, context)
            self._decide_and_act(state, context, plan.reason_summary)
            self.memory.remember(signature, state)
            return self._finish(state)
        except Exception as exc:  # pragma: no cover - top-level safety net
            state.decision = AgentDecision(
                status="error",
                action="suspend_for_operator",
                reason_summary=str(exc),
                alert_required=True,
                suspend_for_human=True,
            )
            state.add_trace("error", "workflow_exception", {"error": str(exc)})
            return self._finish(state)

    def _perceive(self, state: PipeInspectionState, context: ToolContext) -> None:
        state.add_trace("perception", "Tool_Read_Pipe_Text")
        ocr_result = self.tools["Tool_Read_Pipe_Text"].run({"frame_path": state.frame_path}, context)
        if not ocr_result.ok:
            raise RuntimeError(f"OCR tool failed: {ocr_result.error}")
        state.ocr_result = OCRResult(**ocr_result.data)

        parsed = parse_pipe_text(state.ocr_result.text)
        state.material_id = parsed["material_id"]
        state.parsed_material = parsed["material"]
        state.parsed_diameter = parsed["diameter"]

        state.add_trace(
            "perception",
            "Tool_Analyze_Shape",
            {"ocr_latency_ms": round(ocr_result.latency_ms, 2), "parsed": parsed},
        )
        shape_result = self.tools["Tool_Analyze_Shape"].run({"frame_path": state.frame_path}, context)
        if not shape_result.ok:
            raise RuntimeError(f"shape tool failed: {shape_result.error}")
        state.shape_result = ShapeResult(**shape_result.data)

    def _validate_with_erp(self, state: PipeInspectionState, context: ToolContext) -> None:
        if not state.material_id:
            raise RuntimeError("OCR did not produce a usable material_id")
        state.add_trace("validation", "Tool_Query_ERP", {"material_id": state.material_id})
        erp_result = self.tools["Tool_Query_ERP"].run(
            {"material_id": state.material_id, "workstation": state.workstation},
            context,
        )
        if not erp_result.ok:
            raise RuntimeError(f"ERP tool failed: {erp_result.error}")
        state.erp_record = ERPRecord(**erp_result.data)

    def _decide_and_act(self, state: PipeInspectionState, context: ToolContext, reason_summary: str) -> None:
        decision = self._compare_with_bom(state, reason_summary)
        state.decision = decision
        state.add_trace("decision", "conditional_edge", {"decision": decision.status})

        if decision.status == "matched":
            action_result = self.tools["Tool_Prepare_Assembly_Simulation"].run(
                {
                    "run_id": state.run_id,
                    "component_id": state.component_id,
                    "material_id": state.material_id,
                    "workstation": state.workstation,
                },
                context,
            )
            state.add_trace("action", "Tool_Prepare_Assembly_Simulation", action_result.data)
            return

        if decision.alert_required:
            alert_result = self.tools["Tool_Trigger_Alert"].run(
                {
                    "title": "Pipe BOM mismatch",
                    "message": decision.reason_summary,
                    "details": {
                        "run_id": state.run_id,
                        "component_id": state.component_id,
                        "actual_material_id": state.material_id,
                        "expected_material_id": state.erp_record.material_id if state.erp_record else "",
                        "workstation": state.workstation,
                    },
                },
                context,
            )
            state.add_trace("action", "Tool_Trigger_Alert", alert_result.data)

    def _compare_with_bom(self, state: PipeInspectionState, reason_summary: str) -> AgentDecision:
        ocr_confidence = state.ocr_result.confidence if state.ocr_result else 0.0
        if ocr_confidence < self.config.min_ocr_confidence:
            return AgentDecision(
                status="needs_review",
                action="suspend_for_human_review",
                reason_summary=(
                    f"OCR confidence {ocr_confidence:.2f} is below threshold "
                    f"{self.config.min_ocr_confidence:.2f}; manual review is required."
                ),
                alert_required=True,
                suspend_for_human=True,
            )

        expected = state.erp_record
        if not expected:
            return AgentDecision(
                status="needs_review",
                action="suspend_for_human_review",
                reason_summary="No ERP/BOM record was available for validation.",
                alert_required=True,
                suspend_for_human=True,
            )

        mismatches = []
        if normalize_material_id(expected.material_id) != normalize_material_id(state.material_id):
            mismatches.append(
                f"material_id actual={state.material_id}, expected={expected.material_id}"
            )
        if expected.material and state.parsed_material:
            if expected.material.upper() != state.parsed_material.upper():
                mismatches.append(
                    f"material actual={state.parsed_material}, expected={expected.material}"
                )
        if expected.nominal_diameter and state.parsed_diameter:
            if expected.nominal_diameter.upper() != state.parsed_diameter.upper():
                mismatches.append(
                    f"diameter actual={state.parsed_diameter}, expected={expected.nominal_diameter}"
                )

        if mismatches:
            return AgentDecision(
                status="mismatch",
                action="trigger_alert_and_suspend",
                reason_summary=f"{reason_summary} Mismatch found: {'; '.join(mismatches)}.",
                alert_required=True,
                suspend_for_human=True,
            )

        return AgentDecision(
            status="matched",
            action="prepare_assembly_simulation",
            reason_summary=f"{reason_summary} BOM validation passed.",
            alert_required=False,
            suspend_for_human=False,
        )

    def _signature(self, state: PipeInspectionState) -> str:
        if state.component_id and not state.component_id.startswith("component-"):
            return f"{state.workstation}:{state.component_id}"
        shape = state.shape_result.label if state.shape_result else "unknown"
        return f"{state.workstation}:{normalize_material_id(state.material_id)}:{shape}"

    def _finish(self, state: PipeInspectionState) -> PipeInspectionState:
        state.finished_at = now_iso()
        if state.phase not in {"finished", "error"}:
            state.add_trace("finished", "persist_state")
        self.memory.persist_state(state)
        return state

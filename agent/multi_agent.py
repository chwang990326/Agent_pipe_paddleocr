"""Multi-agent orchestration for pipe inspection."""

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
    ProcessChangeReviewResult,
    RetrievedDocument,
    SemanticCorrectionResult,
    ShapeResult,
    new_id,
    now_iso,
    to_plain_data,
)
from agent.tools.base import ToolContext


class PerceptionAgent:
    """Owns visual perception: OCR, shape recognition, and initial parsing."""

    name = "PerceptionAgent"

    def __init__(self, tools: dict[str, object]):
        self.tools = tools

    def run(self, state: PipeInspectionState, context: ToolContext) -> None:
        state.add_trace("perception", "PerceptionAgent.start")

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

        state.add_trace(
            "perception",
            "PerceptionAgent.finish",
            {
                "material_id": state.material_id,
                "shape_label": state.shape_result.label,
            },
        )


class QualityAgent:
    """Owns ERP/BOM validation and semantic OCR correction."""

    name = "QualityAgent"

    def __init__(self, config: AgentConfig, tools: dict[str, object]):
        self.config = config
        self.tools = tools

    def run(self, state: PipeInspectionState, context: ToolContext) -> AgentDecision:
        state.add_trace("validation", "QualityAgent.start")
        self._validate_with_erp(state, context)
        self._semantic_correct_when_needed(state, context)
        decision = self._compare_with_bom(state)
        state.add_trace(
            "validation",
            "QualityAgent.finish",
            {
                "decision": decision.status,
                "material_id": state.material_id,
                "erp_material_id": state.erp_record.material_id if state.erp_record else "",
            },
        )
        return decision

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

    def _semantic_correct_when_needed(self, state: PipeInspectionState, context: ToolContext) -> None:
        if not self.config.semantic_correction_enabled:
            return

        if not self._needs_semantic_correction(state):
            return

        if not self.config.llm_endpoint:
            message = "Semantic OCR correction was triggered but AGENT_LLM_ENDPOINT is not configured"
            if self.config.semantic_correction_required:
                raise RuntimeError(message)
            state.add_trace(
                "reasoning",
                "semantic_correction_skipped",
                {"reason": message, "required": False},
            )
            return

        candidate_records = self._candidate_records(state)
        state.add_trace(
            "reasoning",
            "Tool_Semantic_OCR_Correction",
            {
                "raw_text": state.ocr_result.text if state.ocr_result else "",
                "trigger": self._semantic_correction_trigger(state),
                "candidate_count": len(candidate_records),
            },
        )
        correction_result = self.tools["Tool_Semantic_OCR_Correction"].run(
            {
                "raw_text": state.ocr_result.text if state.ocr_result else "",
                "ocr_confidence": state.ocr_result.confidence if state.ocr_result else 0.0,
                "workstation": state.workstation,
                "task": state.task,
                "candidate_records": candidate_records,
                "shape_context": {
                    "label": state.shape_result.label if state.shape_result else None,
                    "diameter": state.shape_result.diameter if state.shape_result else None,
                    "flange_type": state.shape_result.flange_type if state.shape_result else None,
                },
            },
            context,
        )
        if not correction_result.ok:
            raise RuntimeError(f"semantic OCR correction failed: {correction_result.error}")

        state.semantic_correction = SemanticCorrectionResult(**correction_result.data)
        state.add_trace(
            "reasoning",
            "semantic_correction_result",
            {
                "applied": state.semantic_correction.applied,
                "corrected_text": state.semantic_correction.corrected_text,
                "confidence": state.semantic_correction.confidence,
                "reason_summary": state.semantic_correction.reason_summary,
            },
        )

        if not self._can_apply_correction(state.semantic_correction):
            return

        parsed = parse_pipe_text(state.semantic_correction.corrected_text)
        state.material_id = parsed["material_id"]
        state.parsed_material = parsed["material"]
        state.parsed_diameter = parsed["diameter"]
        state.add_trace("reasoning", "semantic_correction_applied", {"parsed": parsed})
        self._validate_with_erp(state, context)

    def _compare_with_bom(self, state: PipeInspectionState) -> AgentDecision:
        ocr_confidence = state.ocr_result.confidence if state.ocr_result else 0.0
        if ocr_confidence < self.config.min_ocr_confidence and not self._applied_semantic_correction(state):
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
            mismatches.append(f"material_id actual={state.material_id}, expected={expected.material_id}")
        if expected.material and state.parsed_material:
            if expected.material.upper() != state.parsed_material.upper():
                mismatches.append(f"material actual={state.parsed_material}, expected={expected.material}")
        if expected.nominal_diameter and state.parsed_diameter:
            if expected.nominal_diameter.upper() != state.parsed_diameter.upper():
                mismatches.append(
                    f"diameter actual={state.parsed_diameter}, expected={expected.nominal_diameter}"
                )

        if mismatches:
            return AgentDecision(
                status="mismatch",
                action="trigger_alert_and_suspend",
                reason_summary=f"BOM validation failed: {'; '.join(mismatches)}.",
                alert_required=True,
                suspend_for_human=True,
            )

        return AgentDecision(
            status="matched",
            action="prepare_assembly_simulation",
            reason_summary="BOM validation passed.",
            alert_required=False,
            suspend_for_human=False,
        )

    def _needs_semantic_correction(self, state: PipeInspectionState) -> bool:
        ocr_confidence = state.ocr_result.confidence if state.ocr_result else 0.0
        return ocr_confidence < self.config.min_ocr_confidence or self._erp_exact_match(state) is False

    def _semantic_correction_trigger(self, state: PipeInspectionState) -> str:
        triggers = []
        ocr_confidence = state.ocr_result.confidence if state.ocr_result else 0.0
        if ocr_confidence < self.config.min_ocr_confidence:
            triggers.append("low_ocr_confidence")
        if self._erp_exact_match(state) is False:
            triggers.append("erp_material_not_found")
        return ",".join(triggers)

    def _erp_exact_match(self, state: PipeInspectionState) -> bool | None:
        if not state.erp_record:
            return None
        bom_match = state.erp_record.raw.get("bom_match")
        return bom_match if isinstance(bom_match, bool) else None

    def _candidate_records(self, state: PipeInspectionState) -> list[dict]:
        if not state.erp_record:
            return []
        candidates = state.erp_record.raw.get("candidate_records")
        if isinstance(candidates, list):
            return [item for item in candidates if isinstance(item, dict)]
        return [state.erp_record.raw] if state.erp_record.raw else []

    def _can_apply_correction(self, correction: SemanticCorrectionResult) -> bool:
        return (
            correction.applied
            and correction.confidence >= self.config.semantic_correction_min_confidence
            and normalize_material_id(correction.corrected_text) != normalize_material_id(correction.original_text)
        )

    def _applied_semantic_correction(self, state: PipeInspectionState) -> bool:
        correction = state.semantic_correction
        return bool(correction and self._can_apply_correction(correction))


class RAGAgent:
    """Owns process-change document retrieval and LLM review."""

    name = "RAGAgent"

    def __init__(self, config: AgentConfig, tools: dict[str, object]):
        self.config = config
        self.tools = tools

    def run(
        self,
        state: PipeInspectionState,
        context: ToolContext,
        decision: AgentDecision,
    ) -> AgentDecision:
        if decision.status != "matched":
            return decision

        review = self._review_process_changes(state, context)
        if self._process_review_blocks_release(review):
            return AgentDecision(
                status="blocked_by_process_change",
                action=review.action,
                reason_summary=review.reason_summary,
                alert_required=True,
                suspend_for_human=True,
            )
        return decision

    def _review_process_changes(
        self,
        state: PipeInspectionState,
        context: ToolContext,
    ) -> ProcessChangeReviewResult | None:
        if not self.config.process_rag_enabled:
            return None

        if not self.config.llm_endpoint:
            message = "Process-change RAG review was triggered but AGENT_LLM_ENDPOINT is not configured"
            if self.config.process_rag_required:
                raise RuntimeError(message)
            state.add_trace(
                "reasoning",
                "process_change_rag_skipped",
                {"reason": message, "required": False},
            )
            return None

        state.add_trace(
            "reasoning",
            "Tool_Process_Change_RAG_Check",
            {
                "material_id": state.material_id,
                "material": state.parsed_material,
                "diameter": state.parsed_diameter,
                "workstation": state.workstation,
            },
        )
        rag_result = self.tools["Tool_Process_Change_RAG_Check"].run(
            {
                "task": state.task,
                "workstation": state.workstation,
                "material_id": state.material_id,
                "material": state.parsed_material,
                "diameter": state.parsed_diameter,
                "erp_record": to_plain_data(state.erp_record) if state.erp_record else {},
            },
            context,
        )
        if not rag_result.ok:
            raise RuntimeError(f"process-change RAG review failed: {rag_result.error}")

        payload = dict(rag_result.data)
        payload.pop("retrieved_count", None)
        state.process_change_review = self._process_review_from_payload(payload)
        state.add_trace(
            "reasoning",
            "process_change_rag_result",
            {
                "blocked": state.process_change_review.blocked,
                "action": state.process_change_review.action,
                "confidence": state.process_change_review.confidence,
                "citations": [doc.title for doc in state.process_change_review.citations],
            },
        )
        return state.process_change_review

    def _process_review_from_payload(self, payload: dict) -> ProcessChangeReviewResult:
        citations = []
        for item in payload.get("citations") or []:
            if isinstance(item, RetrievedDocument):
                citations.append(item)
            elif isinstance(item, dict):
                citations.append(RetrievedDocument(**item))
        return ProcessChangeReviewResult(
            blocked=bool(payload.get("blocked")),
            action=str(payload.get("action") or ""),
            confidence=float(payload.get("confidence") or 0.0),
            reason_summary=str(payload.get("reason_summary") or ""),
            citations=citations,
            raw=payload.get("raw") if isinstance(payload.get("raw"), dict) else {},
        )

    def _process_review_blocks_release(self, review: ProcessChangeReviewResult | None) -> bool:
        return bool(review and review.blocked and review.confidence >= self.config.process_rag_min_confidence)


class DispatchAgent:
    """Owns downstream actions: simulation preparation and alerting."""

    name = "DispatchAgent"

    def __init__(self, tools: dict[str, object]):
        self.tools = tools

    def run(self, state: PipeInspectionState, context: ToolContext, decision: AgentDecision) -> None:
        state.decision = decision
        state.add_trace("decision", "DispatchAgent.received_decision", {"decision": decision.status})

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
                    "title": self._alert_title(decision),
                    "message": decision.reason_summary,
                    "details": {
                        "run_id": state.run_id,
                        "component_id": state.component_id,
                        "actual_material_id": state.material_id,
                        "expected_material_id": state.erp_record.material_id if state.erp_record else "",
                        "workstation": state.workstation,
                        "process_change_review": to_plain_data(state.process_change_review)
                        if state.process_change_review
                        else {},
                    },
                },
                context,
            )
            state.add_trace("action", "Tool_Trigger_Alert", alert_result.data)

    @staticmethod
    def _alert_title(decision: AgentDecision) -> str:
        if decision.status == "blocked_by_process_change":
            return "Pipe process-change intervention"
        return "Pipe BOM mismatch"


class SupervisorAgent:
    """Coordinates specialist agents and owns final arbitration."""

    name = "SupervisorAgent"

    def __init__(
        self,
        config: AgentConfig,
        tools: dict[str, object],
        memory: AgentMemory,
        brain: ControllerBrain,
    ):
        self.config = config
        self.memory = memory
        self.brain = brain
        self.perception_agent = PerceptionAgent(tools)
        self.quality_agent = QualityAgent(config, tools)
        self.rag_agent = RAGAgent(config, tools)
        self.dispatch_agent = DispatchAgent(tools)

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
            state.add_trace("triggered", "SupervisorAgent.received_task", {"task": task})
            self.perception_agent.run(state, context)

            plan = self.brain.plan(state)
            state.add_trace(
                "reasoning",
                "SupervisorAgent.plan_created",
                {"steps": plan.steps, "reason_summary": plan.reason_summary},
            )

            decision = self.quality_agent.run(state, context)
            decision.reason_summary = f"{plan.reason_summary} {decision.reason_summary}"

            signature = self._signature(state)
            if self.memory.is_duplicate(signature):
                state.decision = AgentDecision(
                    status="duplicate",
                    action="skip_duplicate_component",
                    reason_summary="The current component signature already exists in short-term memory.",
                )
                state.add_trace("finished", "SupervisorAgent.duplicate_skipped", {"signature": signature})
                return self._finish(state)

            decision = self.rag_agent.run(state, context, decision)
            self.dispatch_agent.run(state, context, decision)
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
            state.add_trace("error", "SupervisorAgent.workflow_exception", {"error": str(exc)})
            return self._finish(state)

    def _signature(self, state: PipeInspectionState) -> str:
        if state.component_id and not state.component_id.startswith("component-"):
            return f"{state.workstation}:{state.component_id}"
        shape = state.shape_result.label if state.shape_result else "unknown"
        return f"{state.workstation}:{normalize_material_id(state.material_id)}:{shape}"

    def _finish(self, state: PipeInspectionState) -> PipeInspectionState:
        state.finished_at = now_iso()
        if state.phase not in {"finished", "error"}:
            state.add_trace("finished", "SupervisorAgent.persist_state")
        self.memory.persist_state(state)
        return state

"""Controller brain for planning and auditable reasoning summaries."""

from __future__ import annotations

from dataclasses import dataclass

from agent.schemas import PipeInspectionState


@dataclass
class BrainPlan:
    steps: list[str]
    reason_summary: str
    requires_erp_validation: bool = True


class ControllerBrain:
    """Small deterministic brain with a future LLM adapter boundary.

    In production, this class is the right place to call a local Qwen service.
    The default implementation stays deterministic so the Agent state machine
    can be tested without GPU or network dependencies.
    """

    role = "industrial assembly quality-control and dispatch agent"

    def plan(self, state: PipeInspectionState) -> BrainPlan:
        text = state.ocr_result.text if state.ocr_result else ""
        shape = state.shape_result.label if state.shape_result else "unknown"
        summary = (
            f"Recognized component mark '{text}' with shape '{shape}'. "
            "ERP/BOM validation is required before releasing the component to assembly."
        )
        return BrainPlan(
            steps=[
                "read_pipe_text",
                "analyze_shape",
                "semantic_ocr_correction_when_needed",
                "query_erp",
                "decide_match_or_alert",
            ],
            reason_summary=summary,
            requires_erp_validation=True,
        )

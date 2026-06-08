"""Typed data structures shared by the Agent modules."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass
class TraceEvent:
    timestamp: str
    phase: str
    action: str
    observation: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRResult:
    text: str
    confidence: float
    source: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShapeResult:
    label: str
    confidence: float
    diameter: str | None = None
    flange_type: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ERPRecord:
    material_id: str
    material: str | None = None
    nominal_diameter: str | None = None
    standard: str | None = None
    workstation: str | None = None
    batch_id: str | None = None
    expected_quantity: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticCorrectionResult:
    original_text: str
    corrected_text: str
    confidence: float
    applied: bool
    source: str
    reason_summary: str
    candidates_considered: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedDocument:
    document_id: str
    title: str
    path: str
    score: float
    snippet: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessChangeReviewResult:
    blocked: bool
    action: str
    confidence: float
    reason_summary: str
    citations: list[RetrievedDocument] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDecision:
    status: str
    action: str
    reason_summary: str
    alert_required: bool = False
    suspend_for_human: bool = False


@dataclass
class PipeInspectionState:
    run_id: str
    task: str
    workstation: str
    component_id: str
    batch_id: str
    frame_path: str | None = None
    phase: str = "created"
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    ocr_result: OCRResult | None = None
    shape_result: ShapeResult | None = None
    material_id: str | None = None
    parsed_material: str | None = None
    parsed_diameter: str | None = None
    erp_record: ERPRecord | None = None
    semantic_correction: SemanticCorrectionResult | None = None
    process_change_review: ProcessChangeReviewResult | None = None
    decision: AgentDecision | None = None
    trace: list[TraceEvent] = field(default_factory=list)

    def add_trace(self, phase: str, action: str, observation: dict[str, Any] | None = None) -> None:
        self.phase = phase
        self.trace.append(
            TraceEvent(
                timestamp=now_iso(),
                phase=phase,
                action=action,
                observation=observation or {},
            )
        )


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_plain_data(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {str(k): to_plain_data(v) for k, v in value.items()}
    return value

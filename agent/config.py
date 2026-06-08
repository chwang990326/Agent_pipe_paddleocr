"""Runtime configuration for the Agent project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _text_guard_from_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    if value in {r"\t", "TAB", "tab"}:
        return "\t"
    return value


@dataclass(frozen=True)
class AgentConfig:
    """Configuration loaded from environment variables."""

    project_root: Path
    runtime_dir: Path
    bom_file: Path
    ocr_endpoint: str | None
    shape_endpoint: str | None
    erp_endpoint: str | None
    alert_webhook: str | None
    alert_channel: str
    alert_dry_run: bool
    simulation_endpoint: str | None
    llm_endpoint: str | None
    llm_model: str
    llm_api_key: str | None
    semantic_correction_enabled: bool
    semantic_correction_required: bool
    semantic_correction_min_confidence: float
    http_timeout_seconds: float
    min_ocr_confidence: float
    mock_ocr_text: str
    mock_ocr_confidence: float
    mock_shape_label: str
    mock_shape_confidence: float
    csv_text_guard: str

    @classmethod
    def from_env(cls) -> "AgentConfig":
        project_root = Path(
            os.getenv("AGENT_PROJECT_ROOT", Path(__file__).resolve().parents[1])
        ).resolve()
        runtime_dir = Path(os.getenv("AGENT_RUNTIME_DIR", project_root / "runtime")).resolve()
        bom_file = Path(os.getenv("AGENT_BOM_FILE", project_root / "data" / "bom.sample.json")).resolve()
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        llm_endpoint = (
            os.getenv("AGENT_LLM_ENDPOINT")
            or os.getenv("DEEPSEEK_BASE_URL")
            or ("https://api.deepseek.com" if deepseek_api_key else None)
        )
        llm_model = os.getenv("AGENT_LLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash"

        return cls(
            project_root=project_root,
            runtime_dir=runtime_dir,
            bom_file=bom_file,
            ocr_endpoint=os.getenv("AGENT_OCR_ENDPOINT"),
            shape_endpoint=os.getenv("AGENT_SHAPE_ENDPOINT"),
            erp_endpoint=os.getenv("AGENT_ERP_ENDPOINT"),
            alert_webhook=os.getenv("AGENT_ALERT_WEBHOOK"),
            alert_channel=os.getenv("AGENT_ALERT_CHANNEL", "generic").strip().lower(),
            alert_dry_run=_bool_from_env("AGENT_ALERT_DRY_RUN", True),
            simulation_endpoint=os.getenv("AGENT_SIMULATION_ENDPOINT"),
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
            llm_api_key=os.getenv("AGENT_LLM_API_KEY") or deepseek_api_key,
            semantic_correction_enabled=_bool_from_env("AGENT_SEMANTIC_CORRECTION_ENABLED", True),
            semantic_correction_required=_bool_from_env("AGENT_SEMANTIC_CORRECTION_REQUIRED", False),
            semantic_correction_min_confidence=_float_from_env(
                "AGENT_SEMANTIC_CORRECTION_MIN_CONFIDENCE",
                0.70,
            ),
            http_timeout_seconds=_float_from_env("AGENT_HTTP_TIMEOUT_SECONDS", 10.0),
            min_ocr_confidence=_float_from_env("AGENT_MIN_OCR_CONFIDENCE", 0.75),
            mock_ocr_text=os.getenv("AGENT_MOCK_OCR_TEXT", "304L-DN500"),
            mock_ocr_confidence=_float_from_env("AGENT_MOCK_OCR_CONFIDENCE", 0.85),
            mock_shape_label=os.getenv("AGENT_MOCK_SHAPE_LABEL", "pipe"),
            mock_shape_confidence=_float_from_env("AGENT_MOCK_SHAPE_CONFIDENCE", 0.82),
            csv_text_guard=_text_guard_from_env("AGENT_CSV_TEXT_GUARD", "\t"),
        )

    @property
    def trace_file(self) -> Path:
        return self.runtime_dir / "logs" / "agent_trace.jsonl"

    @property
    def report_file(self) -> Path:
        return self.runtime_dir / "reports" / "batch_report.csv"

    @property
    def short_term_file(self) -> Path:
        return self.runtime_dir / "memory" / "short_term.json"

    @property
    def upload_dir(self) -> Path:
        return self.runtime_dir / "uploads"

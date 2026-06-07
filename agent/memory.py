"""Short-term memory and persistent reporting for Agent runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.config import AgentConfig
from agent.schemas import PipeInspectionState, now_iso, to_plain_data


@dataclass
class ShortTermEntry:
    signature: str
    component_id: str
    material_id: str | None
    timestamp: str


class AgentMemory:
    """Stores run traces and duplicate-recognition memory."""

    report_fields = [
        "timestamp",
        "run_id",
        "batch_id",
        "component_id",
        "workstation",
        "material_id",
        "erp_material_id",
        "decision_status",
        "action",
        "ocr_confidence",
        "shape_label",
        "reason_summary",
    ]

    def __init__(self, config: AgentConfig):
        self.config = config
        self.config.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.report_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.short_term_file.parent.mkdir(parents=True, exist_ok=True)
        self._short_term = self._load_short_term()

    def is_duplicate(self, signature: str) -> bool:
        return signature in self._short_term

    def remember(self, signature: str, state: PipeInspectionState) -> None:
        self._short_term[signature] = ShortTermEntry(
            signature=signature,
            component_id=state.component_id,
            material_id=state.material_id,
            timestamp=now_iso(),
        )
        self._save_short_term()

    def persist_state(self, state: PipeInspectionState) -> None:
        with self.config.trace_file.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(to_plain_data(state), ensure_ascii=False) + "\n")
        self._append_report(state)

    def _load_short_term(self) -> dict[str, ShortTermEntry]:
        path = self.config.short_term_file
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {
                key: ShortTermEntry(**value)
                for key, value in payload.items()
                if isinstance(value, dict)
            }
        except (json.JSONDecodeError, TypeError):
            return {}

    def _save_short_term(self) -> None:
        payload = {key: entry.__dict__ for key, entry in self._short_term.items()}
        self.config.short_term_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _append_report(self, state: PipeInspectionState) -> None:
        file_exists = self.config.report_file.exists()
        with self.config.report_file.open("a", encoding="utf-8-sig", newline="") as report_file:
            writer = csv.DictWriter(
                report_file,
                fieldnames=self.report_fields,
                quoting=csv.QUOTE_ALL,
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(self._report_row(state))

    def _report_row(self, state: PipeInspectionState) -> dict[str, str]:
        decision = state.decision
        erp = state.erp_record
        ocr = state.ocr_result
        shape = state.shape_result
        row: dict[str, Any] = {
            "timestamp": state.finished_at or now_iso(),
            "run_id": state.run_id,
            "batch_id": state.batch_id,
            "component_id": state.component_id,
            "workstation": state.workstation,
            "material_id": state.material_id,
            "erp_material_id": erp.material_id if erp else "",
            "decision_status": decision.status if decision else "",
            "action": decision.action if decision else "",
            "ocr_confidence": f"{ocr.confidence:.4f}" if ocr else "",
            "shape_label": shape.label if shape else "",
            "reason_summary": decision.reason_summary if decision else "",
        }
        return {key: self._csv_text(value) for key, value in row.items()}

    def _csv_text(self, value: Any) -> str:
        """Force spreadsheet software to treat exported values as plain text."""

        text = "" if value is None else str(value)
        text = text.replace("\r", " ").replace("\n", " ").strip()
        return f"{self.config.csv_text_guard}{text}" if text else ""

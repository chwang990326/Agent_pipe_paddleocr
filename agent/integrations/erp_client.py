"""ERP/MES adapter used by Tool_Query_ERP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.config import AgentConfig
from agent.parsing import normalize_material_id
from agent.schemas import ERPRecord


class ERPClient:
    def __init__(self, config: AgentConfig):
        self.config = config

    def query_expected_record(self, material_id: str, workstation: str) -> ERPRecord:
        if self.config.erp_endpoint:
            return self._query_http(material_id, workstation)
        return self._query_local_bom(material_id, workstation)

    def _query_http(self, material_id: str, workstation: str) -> ERPRecord:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise RuntimeError("requests is required for HTTP ERP calls") from exc

        response = requests.post(
            self.config.erp_endpoint,
            json={"material_id": material_id, "workstation": workstation},
            timeout=self.config.http_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return self._record_from_dict(payload, workstation=workstation)

    def _query_local_bom(self, material_id: str, workstation: str) -> ERPRecord:
        path = Path(self.config.bom_file)
        if not path.exists():
            raise FileNotFoundError(f"BOM file not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        records = data.get("workstations", {}).get(workstation, [])
        if not records:
            raise LookupError(f"no BOM records configured for workstation: {workstation}")

        normalized_actual = normalize_material_id(material_id)
        for item in records:
            if normalize_material_id(item.get("material_id")) == normalized_actual:
                record = self._record_from_dict(item, workstation=workstation)
                record.raw["bom_match"] = True
                return record

        record = self._record_from_dict(records[0], workstation=workstation)
        record.raw["bom_match"] = False
        record.raw["candidate_count"] = len(records)
        return record

    @staticmethod
    def _record_from_dict(payload: dict[str, Any], workstation: str | None = None) -> ERPRecord:
        return ERPRecord(
            material_id=str(payload.get("material_id") or ""),
            material=payload.get("material"),
            nominal_diameter=payload.get("nominal_diameter"),
            standard=payload.get("standard"),
            workstation=payload.get("workstation") or workstation,
            batch_id=payload.get("batch_id"),
            expected_quantity=payload.get("expected_quantity"),
            raw=dict(payload),
        )

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from agent import build_default_agent
from agent.config import AgentConfig


def make_config(tmp_path: Path, mock_text: str = "304L-DN500") -> AgentConfig:
    project_root = Path(__file__).resolve().parents[1]
    runtime_dir = tmp_path / "runtime"
    return AgentConfig(
        project_root=project_root,
        runtime_dir=runtime_dir,
        bom_file=project_root / "data" / "bom.sample.json",
        ocr_endpoint=None,
        shape_endpoint=None,
        erp_endpoint=None,
        alert_webhook=None,
        alert_channel="generic",
        alert_dry_run=True,
        simulation_endpoint=None,
        llm_endpoint=None,
        llm_model="qwen-14b-industrial",
        llm_api_key=None,
        http_timeout_seconds=1.0,
        min_ocr_confidence=0.75,
        mock_ocr_text=mock_text,
        mock_ocr_confidence=0.85,
        mock_shape_label="pipe",
        mock_shape_confidence=0.82,
        csv_text_guard="\t",
    )


class AgentWorkflowTests(unittest.TestCase):
    def test_matched_path_prepares_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp))
            state = build_default_agent(config).inspect_pipe(
                task="inspect",
                workstation="A-01",
                component_id="sensor-001",
            )

            self.assertEqual(state.decision.status, "matched")
            self.assertEqual(state.decision.action, "prepare_assembly_simulation")

    def test_mismatch_path_triggers_alert(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp))
            state = build_default_agent(config).inspect_pipe(
                task="inspect",
                workstation="A-02",
                component_id="sensor-002",
            )

            self.assertEqual(state.decision.status, "mismatch")
            self.assertTrue(state.decision.alert_required)
            self.assertTrue(state.decision.suspend_for_human)

    def test_csv_fields_are_exported_as_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp))
            build_default_agent(config).inspect_pipe(
                task="inspect",
                workstation="A-01",
                component_id="sensor-003",
                batch_id="BATCH-20260607-000000000001",
            )

            with config.report_file.open("r", encoding="utf-8-sig", newline="") as report_file:
                rows = list(csv.DictReader(report_file))

            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["timestamp"].startswith("\t"))
            self.assertTrue(rows[0]["batch_id"].startswith("\t"))
            self.assertTrue(rows[0]["material_id"].startswith("\t"))


if __name__ == "__main__":
    unittest.main()

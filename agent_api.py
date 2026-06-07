"""Flask API wrapper for the Controller Agent."""

from __future__ import annotations

import os
from pathlib import Path

from agent import build_default_agent
from agent.config import AgentConfig
from agent.schemas import to_plain_data

try:
    from flask import Flask, jsonify, request
except ImportError as exc:  # pragma: no cover - import-time guidance
    raise RuntimeError("Flask is required to run agent_api.py") from exc


config = AgentConfig.from_env()
config.upload_dir.mkdir(parents=True, exist_ok=True)
agent = build_default_agent(config)
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "pipe-inspection-agent"})


@app.post("/agent/pipe-inspection")
def pipe_inspection():
    payload = request.get_json(silent=True) or {}
    frame_path = payload.get("frame_path")

    if "file" in request.files:
        uploaded = request.files["file"]
        safe_name = Path(uploaded.filename or "frame.jpg").name
        target = config.upload_dir / safe_name
        uploaded.save(target)
        frame_path = str(target)

    state = agent.inspect_pipe(
        task=payload.get("task") or "Inspect current pipe material preparation.",
        frame_path=frame_path,
        workstation=payload.get("workstation") or "A-01",
        component_id=payload.get("component_id"),
        batch_id=payload.get("batch_id"),
    )
    status_code = 200 if state.decision and state.decision.status in {"matched", "duplicate"} else 409
    return jsonify(to_plain_data(state)), status_code


if __name__ == "__main__":
    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "8091"))
    app.run(host=host, port=port, debug=os.getenv("AGENT_DEBUG", "0") == "1")

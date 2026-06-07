"""Shape analysis adapter."""

from __future__ import annotations

from pathlib import Path

from agent.config import AgentConfig
from agent.schemas import ShapeResult


class ShapeAnalysisClient:
    def __init__(self, config: AgentConfig):
        self.config = config

    def analyze(self, frame_path: str | None) -> ShapeResult:
        if not self.config.shape_endpoint:
            return ShapeResult(
                label=self.config.mock_shape_label,
                confidence=self.config.mock_shape_confidence,
                raw={"frame_path": frame_path, "source": "mock"},
            )

        if not frame_path:
            raise ValueError("frame_path is required when AGENT_SHAPE_ENDPOINT is configured")

        path = Path(frame_path)
        if not path.exists():
            raise FileNotFoundError(f"frame file not found: {path}")

        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise RuntimeError("requests is required for HTTP shape calls") from exc

        with path.open("rb") as image_file:
            response = requests.post(
                self.config.shape_endpoint,
                files={"file": (path.name, image_file)},
                timeout=self.config.http_timeout_seconds,
            )
        response.raise_for_status()
        payload = response.json()
        return ShapeResult(
            label=str(payload.get("label") or payload.get("shape") or "unknown"),
            confidence=float(payload.get("confidence") or 0.0),
            diameter=payload.get("diameter"),
            flange_type=payload.get("flange_type"),
            raw=payload,
        )

"""OCR service adapter.

The production path posts the current frame to the Jetson TX2 PaddleOCR service.
Without an endpoint, the adapter returns a deterministic mock so the Agent flow
can be developed and tested on a laptop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.config import AgentConfig
from agent.schemas import OCRResult


class OCRServiceClient:
    def __init__(self, config: AgentConfig):
        self.config = config

    def read_text(self, frame_path: str | None) -> OCRResult:
        if not self.config.ocr_endpoint:
            return OCRResult(
                text=self.config.mock_ocr_text,
                confidence=self.config.mock_ocr_confidence,
                source="mock",
                raw={"frame_path": frame_path},
            )

        if not frame_path:
            raise ValueError("frame_path is required when AGENT_OCR_ENDPOINT is configured")

        path = Path(frame_path)
        if not path.exists():
            raise FileNotFoundError(f"frame file not found: {path}")

        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise RuntimeError("requests is required for HTTP OCR calls") from exc

        with path.open("rb") as image_file:
            response = requests.post(
                self.config.ocr_endpoint,
                files={"file": (path.name, image_file)},
                timeout=self.config.http_timeout_seconds,
            )
        response.raise_for_status()
        payload = response.json()
        text, confidence = self._extract_text(payload)
        return OCRResult(text=text, confidence=confidence, source=self.config.ocr_endpoint, raw=payload)

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> tuple[str, float]:
        if "text" in payload:
            return str(payload.get("text") or ""), float(payload.get("confidence") or 0.0)

        for key in ("recognized_text", "result_text"):
            if key in payload:
                return str(payload.get(key) or ""), float(payload.get("confidence") or 0.0)

        for key in ("results", "result", "识别结果"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = [str(item) for item in value if str(item).strip()]
                return (max(candidates, key=len) if candidates else "", float(payload.get("confidence") or 0.0))
            if isinstance(value, str):
                return value, float(payload.get("confidence") or 0.0)

        raise ValueError(f"cannot extract OCR text from payload keys: {list(payload.keys())}")

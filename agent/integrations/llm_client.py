"""LLM adapter for semantic OCR correction."""

from __future__ import annotations

import json
from typing import Any

from agent.config import AgentConfig
from agent.parsing import normalize_material_id
from agent.schemas import SemanticCorrectionResult


class SemanticOCRCorrectionClient:
    """Calls an LLM to correct OCR text using industrial context.

    The production path expects an OpenAI-compatible chat-completions service or
    a direct JSON endpoint. The mock path is only for offline tests and demos.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def correct(
        self,
        raw_text: str,
        ocr_confidence: float,
        workstation: str,
        task: str,
        candidate_records: list[dict[str, Any]],
        shape_context: dict[str, Any] | None = None,
    ) -> SemanticCorrectionResult:
        if not self.config.llm_endpoint:
            raise RuntimeError("AGENT_LLM_ENDPOINT is required for semantic OCR correction")

        if self.config.llm_endpoint.startswith("mock://"):
            return self._mock_correct(raw_text, ocr_confidence, candidate_records)

        payload = self._build_payload(
            raw_text=raw_text,
            ocr_confidence=ocr_confidence,
            workstation=workstation,
            task=task,
            candidate_records=candidate_records,
            shape_context=shape_context or {},
        )
        response_payload = self._post_to_llm(payload)
        parsed = self._extract_json(response_payload)
        return self._result_from_payload(parsed, raw_text, candidate_records)

    def _post_to_llm(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise RuntimeError("requests is required for HTTP LLM calls") from exc

        endpoint = self.config.llm_endpoint.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"

        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=self.config.http_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _build_payload(
        self,
        raw_text: str,
        ocr_confidence: float,
        workstation: str,
        task: str,
        candidate_records: list[dict[str, Any]],
        shape_context: dict[str, Any],
    ) -> dict[str, Any]:
        system_prompt = (
            "You are an industrial semantic OCR correction engine for metallurgy "
            "and shipbuilding assembly inspection. Correct OCR text only when the "
            "industrial context, BOM candidates, material standards, and visual "
            "character similarity support the correction. Return JSON only."
        )
        user_payload = {
            "raw_ocr_text": raw_text,
            "ocr_confidence": ocr_confidence,
            "workstation": workstation,
            "task": task,
            "shape_context": shape_context,
            "bom_candidates": candidate_records,
            "domain_rules": [
                "Q345B, Q235B, 304L, and 316L are common material grades.",
                "DN followed by digits denotes nominal pipe diameter.",
                "OCR often confuses O and 0, Q and 0, I/l and 1, S and 5, B and 8.",
                "Never invent a material that is unsupported by BOM candidates or context.",
            ],
            "required_output_schema": {
                "applied": "boolean",
                "corrected_text": "string",
                "confidence": "number between 0 and 1",
                "reason_summary": "short auditable explanation",
                "candidates_considered": ["material ids"],
            },
        }
        return {
            "model": self.config.llm_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }

    def _extract_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "corrected_text" in payload or "applied" in payload:
            return payload

        choices = payload.get("choices")
        if choices:
            content = choices[0].get("message", {}).get("content", "")
        else:
            content = payload.get("content", "")

        if isinstance(content, dict):
            return content

        text = str(content).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise ValueError(f"LLM response did not contain valid JSON: {text[:200]}")

    def _result_from_payload(
        self,
        payload: dict[str, Any],
        raw_text: str,
        candidate_records: list[dict[str, Any]],
    ) -> SemanticCorrectionResult:
        corrected_text = normalize_material_id(payload.get("corrected_text") or raw_text)
        candidates = payload.get("candidates_considered")
        if not isinstance(candidates, list):
            candidates = [str(item.get("material_id")) for item in candidate_records if item.get("material_id")]
        confidence = float(payload.get("confidence") or 0.0)
        confidence = max(0.0, min(confidence, 1.0))
        applied = bool(payload.get("applied")) and corrected_text != normalize_material_id(raw_text)

        return SemanticCorrectionResult(
            original_text=raw_text,
            corrected_text=corrected_text,
            confidence=confidence,
            applied=applied,
            source=self.config.llm_endpoint or "",
            reason_summary=str(payload.get("reason_summary") or ""),
            candidates_considered=[str(item) for item in candidates],
            raw=payload,
        )

    def _mock_correct(
        self,
        raw_text: str,
        ocr_confidence: float,
        candidate_records: list[dict[str, Any]],
    ) -> SemanticCorrectionResult:
        raw_key = self._visual_key(raw_text)
        candidate_ids = [
            normalize_material_id(item.get("material_id"))
            for item in candidate_records
            if item.get("material_id")
        ]
        for candidate in candidate_ids:
            if self._visual_key(candidate) == raw_key:
                return SemanticCorrectionResult(
                    original_text=raw_text,
                    corrected_text=candidate,
                    confidence=max(0.86, ocr_confidence),
                    applied=candidate != normalize_material_id(raw_text),
                    source=self.config.llm_endpoint or "mock://semantic-correction",
                    reason_summary=(
                        "Mock LLM correction: visual confusion pattern matches a BOM "
                        "candidate and follows pipe material/DN naming conventions."
                    ),
                    candidates_considered=candidate_ids,
                    raw={"mode": "mock"},
                )

        return SemanticCorrectionResult(
            original_text=raw_text,
            corrected_text=normalize_material_id(raw_text),
            confidence=ocr_confidence,
            applied=False,
            source=self.config.llm_endpoint or "mock://semantic-correction",
            reason_summary="Mock LLM correction found no safe domain-supported correction.",
            candidates_considered=candidate_ids,
            raw={"mode": "mock"},
        )

    @staticmethod
    def _visual_key(text: str | None) -> str:
        value = normalize_material_id(text)
        return (
            value.replace("O", "0")
            .replace("Q", "0")
            .replace("I", "1")
            .replace("L", "1")
            .replace("S", "5")
        )

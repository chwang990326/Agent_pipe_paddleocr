"""Agentic RAG review for dynamic process-change documents."""

from __future__ import annotations

import json
from typing import Any

from agent.config import AgentConfig
from agent.schemas import ProcessChangeReviewResult, RetrievedDocument


class ProcessChangeRAGClient:
    """Uses retrieved process documents and an LLM to review release decisions."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def review(
        self,
        task: str,
        workstation: str,
        material_id: str | None,
        material: str | None,
        diameter: str | None,
        erp_record: dict[str, Any],
        retrieved_documents: list[dict[str, Any]],
    ) -> ProcessChangeReviewResult:
        citations = [self._document_from_dict(item) for item in retrieved_documents]
        if not retrieved_documents:
            return ProcessChangeReviewResult(
                blocked=False,
                action="no_relevant_process_change",
                confidence=1.0,
                reason_summary="No relevant process-change documents were retrieved.",
                citations=[],
                raw={"retrieved_count": 0},
            )

        if not self.config.llm_endpoint:
            raise RuntimeError("AGENT_LLM_ENDPOINT is required for process-change RAG review")

        if self.config.llm_endpoint.startswith("mock://"):
            return self._mock_review(material_id, material, retrieved_documents, citations)

        payload = self._build_payload(
            task=task,
            workstation=workstation,
            material_id=material_id,
            material=material,
            diameter=diameter,
            erp_record=erp_record,
            retrieved_documents=retrieved_documents,
        )
        response_payload = self._post_to_llm(payload)
        parsed = self._extract_json(response_payload)
        return self._result_from_payload(parsed, citations)

    def _post_to_llm(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise RuntimeError("requests is required for HTTP LLM calls") from exc

        endpoint = self._chat_completions_url()
        headers = {"Content-Type": "application/json"}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"
        elif self._is_deepseek_endpoint(endpoint):
            raise RuntimeError("DEEPSEEK_API_KEY or AGENT_LLM_API_KEY is required for DeepSeek calls")

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
        task: str,
        workstation: str,
        material_id: str | None,
        material: str | None,
        diameter: str | None,
        erp_record: dict[str, Any],
        retrieved_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system_prompt = (
            "You are an industrial process-change review agent for shipbuilding "
            "quality inspection. Read retrieved process documents and decide whether "
            "the current pipe can be released after BOM validation. Return JSON only. "
            "Do not block unless the document explicitly applies to the material, "
            "workstation, batch, shipyard area, or task context."
        )
        user_payload = {
            "task": task,
            "workstation": workstation,
            "recognized_material": {
                "material_id": material_id,
                "material": material,
                "diameter": diameter,
            },
            "erp_record": erp_record,
            "retrieved_documents": retrieved_documents,
            "required_output_schema": {
                "blocked": "boolean",
                "action": "release | send_to_reinspection | suspend_for_human_review | no_relevant_process_change",
                "confidence": "number between 0 and 1",
                "reason_summary": "short explanation with document title",
                "citation_ids": ["document ids"],
            },
        }
        return {
            "model": self.config.llm_model,
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": 700,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }

    def _result_from_payload(
        self,
        payload: dict[str, Any],
        citations: list[RetrievedDocument],
    ) -> ProcessChangeReviewResult:
        confidence = float(payload.get("confidence") or 0.0)
        confidence = max(0.0, min(confidence, 1.0))
        citation_ids = payload.get("citation_ids")
        if isinstance(citation_ids, list):
            selected = {str(item) for item in citation_ids}
            cited = [doc for doc in citations if doc.document_id in selected] or citations
        else:
            cited = citations

        return ProcessChangeReviewResult(
            blocked=bool(payload.get("blocked")),
            action=str(payload.get("action") or "suspend_for_human_review"),
            confidence=confidence,
            reason_summary=str(payload.get("reason_summary") or ""),
            citations=cited,
            raw=payload,
        )

    def _mock_review(
        self,
        material_id: str | None,
        material: str | None,
        retrieved_documents: list[dict[str, Any]],
        citations: list[RetrievedDocument],
    ) -> ProcessChangeReviewResult:
        material_tokens = {
            item.upper()
            for item in [material_id, material]
            if item
        }
        for doc in retrieved_documents:
            text = f"{doc.get('title', '')}\n{doc.get('snippet', '')}".upper()
            applies = any(token and token in text for token in material_tokens)
            blocks = any(term in text for term in ["禁止直接装配", "禁止装配", "复检", "暂停装配"])
            if applies and blocks:
                return ProcessChangeReviewResult(
                    blocked=True,
                    action="send_to_reinspection",
                    confidence=0.88,
                    reason_summary=(
                        f"Mock RAG review: {doc.get('title')} applies to {material_id} "
                        "and requires reinspection before assembly."
                    ),
                    citations=citations,
                    raw={"mode": "mock", "matched_document_id": doc.get("document_id")},
                )

        return ProcessChangeReviewResult(
            blocked=False,
            action="release",
            confidence=0.82,
            reason_summary="Mock RAG review found no applicable blocking process change.",
            citations=citations,
            raw={"mode": "mock"},
        )

    def _extract_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "blocked" in payload or "action" in payload:
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

    def _chat_completions_url(self) -> str:
        endpoint = self.config.llm_endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            return endpoint
        if self._is_deepseek_endpoint(endpoint):
            return f"{endpoint}/chat/completions"
        return f"{endpoint}/v1/chat/completions"

    @staticmethod
    def _is_deepseek_endpoint(endpoint: str) -> bool:
        return "api.deepseek.com" in endpoint.lower()

    @staticmethod
    def _document_from_dict(payload: dict[str, Any]) -> RetrievedDocument:
        return RetrievedDocument(
            document_id=str(payload.get("document_id") or ""),
            title=str(payload.get("title") or ""),
            path=str(payload.get("path") or ""),
            score=float(payload.get("score") or 0.0),
            snippet=str(payload.get("snippet") or ""),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )

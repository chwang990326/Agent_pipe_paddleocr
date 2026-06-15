"""Multi-round LLM client for production incident investigation."""

from __future__ import annotations

import json
from typing import Any

from agent.config import AgentConfig
from agent.schemas import IncidentInvestigationResult


class IncidentInvestigationClient:
    """Runs summary, hypothesis, and disposition rounds with an LLM."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def investigate(
        self,
        current_event: dict[str, Any],
        recent_events: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> IncidentInvestigationResult:
        if not self.config.llm_endpoint:
            raise RuntimeError("AGENT_LLM_ENDPOINT is required for incident investigation")

        if self.config.llm_endpoint.startswith("mock://"):
            return self._mock_investigate(current_event, recent_events, evidence)

        incident_summary = self._round_summary(current_event, recent_events)
        hypotheses = self._round_hypotheses(incident_summary, evidence)
        final = self._round_final(incident_summary, hypotheses, evidence)
        return self._result_from_payload(final, incident_summary, hypotheses, evidence)

    def _round_summary(self, current_event: dict[str, Any], recent_events: list[dict[str, Any]]) -> str:
        payload = self._chat_payload(
            system_prompt=(
                "You summarize recent industrial inspection events. Compact them "
                "into a short incident timeline. Return JSON only."
            ),
            user_payload={
                "current_event": current_event,
                "recent_events": recent_events,
                "required_output_schema": {"incident_summary": "short string"},
            },
            max_tokens=400,
        )
        data = self._extract_json(self._post_to_llm(payload))
        return str(data.get("incident_summary") or "")

    def _round_hypotheses(self, incident_summary: str, evidence: list[dict[str, Any]]) -> list[str]:
        payload = self._chat_payload(
            system_prompt=(
                "You propose root-cause hypotheses for repeated industrial anomalies. "
                "Return JSON only."
            ),
            user_payload={
                "incident_summary": incident_summary,
                "evidence": evidence,
                "required_output_schema": {"hypotheses": ["short strings"]},
            },
            max_tokens=600,
        )
        data = self._extract_json(self._post_to_llm(payload))
        hypotheses = data.get("hypotheses")
        if not isinstance(hypotheses, list):
            return []
        return [str(item) for item in hypotheses]

    def _round_final(
        self,
        incident_summary: str,
        hypotheses: list[str],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = self._chat_payload(
            system_prompt=(
                "You are a production incident investigation agent. Choose the most "
                "likely root cause and recommend safe actions. Return JSON only."
            ),
            user_payload={
                "incident_summary": incident_summary,
                "hypotheses": hypotheses,
                "evidence": evidence,
                "required_output_schema": {
                    "triggered": "boolean",
                    "severity": "low | medium | high",
                    "confidence": "number between 0 and 1",
                    "root_cause_summary": "short string",
                    "recommended_actions": ["short strings"],
                },
            },
            max_tokens=800,
        )
        return self._extract_json(self._post_to_llm(payload))

    def _chat_payload(self, system_prompt: str, user_payload: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        return {
            "model": self.config.llm_model,
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }

    def _post_to_llm(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
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

    def _result_from_payload(
        self,
        payload: dict[str, Any],
        incident_summary: str,
        hypotheses: list[str],
        evidence: list[dict[str, Any]],
    ) -> IncidentInvestigationResult:
        confidence = float(payload.get("confidence") or 0.0)
        confidence = max(0.0, min(confidence, 1.0))
        actions = payload.get("recommended_actions")
        if not isinstance(actions, list):
            actions = []
        return IncidentInvestigationResult(
            triggered=bool(payload.get("triggered")),
            severity=str(payload.get("severity") or "medium"),
            confidence=confidence,
            incident_summary=incident_summary,
            root_cause_summary=str(payload.get("root_cause_summary") or ""),
            hypotheses=hypotheses,
            evidence=evidence,
            recommended_actions=[str(item) for item in actions],
            raw=payload,
        )

    def _mock_investigate(
        self,
        current_event: dict[str, Any],
        recent_events: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> IncidentInvestigationResult:
        workstation = current_event.get("workstation", "")
        material_id = current_event.get("material_id", "")
        abnormal_count = len(
            [event for event in [*recent_events, current_event] if event.get("decision_status") not in {"matched", "duplicate", ""}]
        )
        summary = (
            f"{workstation} workstation has {abnormal_count} recent abnormal inspections, "
            f"main material={material_id}."
        )
        return IncidentInvestigationResult(
            triggered=True,
            severity="high",
            confidence=0.88,
            incident_summary=summary,
            root_cause_summary=(
                "Repeated abnormal decisions indicate a production-level issue, "
                "likely a restricted batch, unsynchronized process change, or dispatch mismatch."
            ),
            hypotheses=[
                "Current batch is affected by process-change reinspection requirements.",
                "Warehouse or dispatch is still feeding restricted material to the workstation.",
                "ERP/BOM or process-document synchronization may be lagging.",
            ],
            evidence=evidence,
            recommended_actions=[
                f"Pause {workstation} release for {material_id}.",
                "Route the current batch to reinspection or manual quality review.",
                "Notify dispatch, warehouse, and the production owner.",
                "Verify ERP/BOM and latest process-change documents before resuming.",
            ],
            raw={"mode": "mock", "abnormal_count": abnormal_count},
        )

    def _extract_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        if any(key in payload for key in ("incident_summary", "hypotheses", "root_cause_summary")):
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

"""Alert webhook adapter for Feishu, DingTalk, or a generic HTTP receiver."""

from __future__ import annotations

from agent.config import AgentConfig


class AlertClient:
    def __init__(self, config: AgentConfig):
        self.config = config

    def send(self, title: str, message: str, details: dict) -> dict:
        payload = self._format_payload(title, message, details)

        if self.config.alert_dry_run or not self.config.alert_webhook:
            return {"sent": False, "dry_run": True, "payload": payload}

        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise RuntimeError("requests is required for HTTP alert calls") from exc

        response = requests.post(
            self.config.alert_webhook,
            json=payload,
            timeout=self.config.http_timeout_seconds,
        )
        response.raise_for_status()
        return {"sent": True, "status_code": response.status_code}

    def _format_payload(self, title: str, message: str, details: dict) -> dict:
        text = f"{title}\n{message}\n{details}"
        if self.config.alert_channel == "feishu":
            return {"msg_type": "text", "content": {"text": text}}
        if self.config.alert_channel == "dingtalk":
            return {"msgtype": "text", "text": {"content": text}}
        return {"title": title, "message": message, "details": details}

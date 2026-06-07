"""Assembly simulation adapter."""

from __future__ import annotations

from agent.config import AgentConfig


class AssemblySimulationClient:
    def __init__(self, config: AgentConfig):
        self.config = config

    def prepare(self, payload: dict) -> dict:
        if not self.config.simulation_endpoint:
            return {"prepared": True, "dry_run": True, "payload": payload}

        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise RuntimeError("requests is required for HTTP simulation calls") from exc

        response = requests.post(
            self.config.simulation_endpoint,
            json=payload,
            timeout=self.config.http_timeout_seconds,
        )
        response.raise_for_status()
        return {"prepared": True, "status_code": response.status_code}

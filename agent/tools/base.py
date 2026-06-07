"""Base protocol for Agent tools."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.config import AgentConfig


@dataclass
class ToolContext:
    config: AgentConfig
    run_id: str
    memory: Any | None = None


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0


class AgentTool(Protocol):
    name: str
    description: str

    def run(self, payload: dict[str, Any], context: ToolContext) -> ToolResult:
        ...


def timed_call(fn):
    start = time.perf_counter()
    try:
        data = fn()
        return ToolResult(ok=True, data=data, latency_ms=(time.perf_counter() - start) * 1000)
    except Exception as exc:  # pragma: no cover - defensive tool boundary
        return ToolResult(ok=False, error=str(exc), latency_ms=(time.perf_counter() - start) * 1000)

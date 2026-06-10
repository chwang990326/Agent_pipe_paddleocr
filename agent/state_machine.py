"""Compatibility wrapper for the multi-agent pipe inspection workflow."""

from __future__ import annotations

from agent.brain import ControllerBrain
from agent.config import AgentConfig
from agent.memory import AgentMemory
from agent.multi_agent import SupervisorAgent
from agent.schemas import PipeInspectionState


class PipeInspectionWorkflow:
    """Keeps the public workflow API while delegating to SupervisorAgent."""

    def __init__(
        self,
        config: AgentConfig,
        tools: dict[str, object],
        memory: AgentMemory,
        brain: ControllerBrain,
    ):
        self.config = config
        self.tools = tools
        self.memory = memory
        self.brain = brain
        self.supervisor = SupervisorAgent(
            config=config,
            tools=tools,
            memory=memory,
            brain=brain,
        )

    def run(
        self,
        task: str,
        frame_path: str | None,
        workstation: str,
        component_id: str | None = None,
        batch_id: str | None = None,
    ) -> PipeInspectionState:
        return self.supervisor.run(
            task=task,
            frame_path=frame_path,
            workstation=workstation,
            component_id=component_id,
            batch_id=batch_id,
        )

"""Top-level Controller Agent facade."""

from __future__ import annotations

from agent.brain import ControllerBrain
from agent.config import AgentConfig
from agent.memory import AgentMemory
from agent.schemas import PipeInspectionState
from agent.state_machine import PipeInspectionWorkflow
from agent.tools.registry import build_tool_registry


class ControllerAgent:
    def __init__(self, workflow: PipeInspectionWorkflow):
        self.workflow = workflow

    def inspect_pipe(
        self,
        task: str,
        frame_path: str | None = None,
        workstation: str = "A-01",
        component_id: str | None = None,
        batch_id: str | None = None,
    ) -> PipeInspectionState:
        return self.workflow.run(
            task=task,
            frame_path=frame_path,
            workstation=workstation,
            component_id=component_id,
            batch_id=batch_id,
        )


def build_default_agent(config: AgentConfig | None = None) -> ControllerAgent:
    runtime_config = config or AgentConfig.from_env()
    memory = AgentMemory(runtime_config)
    workflow = PipeInspectionWorkflow(
        config=runtime_config,
        tools=build_tool_registry(),
        memory=memory,
        brain=ControllerBrain(),
    )
    return ControllerAgent(workflow)

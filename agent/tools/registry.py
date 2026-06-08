"""Tool registry for the Controller Agent."""

from __future__ import annotations

from agent.tools.action import PrepareAssemblySimulationTool, QueryERPTool, TriggerAlertTool
from agent.tools.perception import AnalyzeShapeTool, ReadPipeTextTool
from agent.tools.reasoning import SemanticOCRCorrectionTool


def build_tool_registry() -> dict[str, object]:
    tools = [
        ReadPipeTextTool(),
        AnalyzeShapeTool(),
        SemanticOCRCorrectionTool(),
        QueryERPTool(),
        TriggerAlertTool(),
        PrepareAssemblySimulationTool(),
    ]
    return {tool.name: tool for tool in tools}

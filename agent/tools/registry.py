"""Tool registry for the Controller Agent."""

from __future__ import annotations

from agent.tools.action import PrepareAssemblySimulationTool, QueryERPTool, TriggerAlertTool
from agent.tools.perception import AnalyzeShapeTool, ReadPipeTextTool
from agent.tools.reasoning import ProcessChangeRAGCheckTool, SemanticOCRCorrectionTool


def build_tool_registry() -> dict[str, object]:
    tools = [
        ReadPipeTextTool(),
        AnalyzeShapeTool(),
        SemanticOCRCorrectionTool(),
        ProcessChangeRAGCheckTool(),
        QueryERPTool(),
        TriggerAlertTool(),
        PrepareAssemblySimulationTool(),
    ]
    return {tool.name: tool for tool in tools}

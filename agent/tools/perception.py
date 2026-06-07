"""Perception tools: the Agent's eyes."""

from __future__ import annotations

from agent.integrations.ocr_client import OCRServiceClient
from agent.integrations.shape_client import ShapeAnalysisClient
from agent.schemas import to_plain_data
from agent.tools.base import ToolContext, ToolResult, timed_call


class ReadPipeTextTool:
    name = "Tool_Read_Pipe_Text"
    description = "Read handwritten or printed marks on the pipe through OCR."

    def run(self, payload: dict, context: ToolContext) -> ToolResult:
        def _call():
            result = OCRServiceClient(context.config).read_text(payload.get("frame_path"))
            return to_plain_data(result)

        return timed_call(_call)


class AnalyzeShapeTool:
    name = "Tool_Analyze_Shape"
    description = "Analyze physical shape parameters such as diameter and flange type."

    def run(self, payload: dict, context: ToolContext) -> ToolResult:
        def _call():
            result = ShapeAnalysisClient(context.config).analyze(payload.get("frame_path"))
            return to_plain_data(result)

        return timed_call(_call)

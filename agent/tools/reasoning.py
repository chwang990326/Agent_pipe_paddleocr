"""LLM-backed reasoning tools."""

from __future__ import annotations

from agent.integrations.llm_client import SemanticOCRCorrectionClient
from agent.tools.base import ToolContext, ToolResult, timed_call


class SemanticOCRCorrectionTool:
    name = "Tool_Semantic_OCR_Correction"
    description = "Use an LLM to correct OCR text with metallurgy and shipbuilding context."

    def run(self, payload: dict, context: ToolContext) -> ToolResult:
        def _call():
            result = SemanticOCRCorrectionClient(context.config).correct(
                raw_text=payload["raw_text"],
                ocr_confidence=float(payload.get("ocr_confidence") or 0.0),
                workstation=payload.get("workstation", ""),
                task=payload.get("task", ""),
                candidate_records=payload.get("candidate_records") or [],
                shape_context=payload.get("shape_context") or {},
            )
            return {
                "original_text": result.original_text,
                "corrected_text": result.corrected_text,
                "confidence": result.confidence,
                "applied": result.applied,
                "source": result.source,
                "reason_summary": result.reason_summary,
                "candidates_considered": result.candidates_considered,
                "raw": result.raw,
            }

        return timed_call(_call)

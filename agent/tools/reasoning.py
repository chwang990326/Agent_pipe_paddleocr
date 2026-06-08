"""LLM-backed reasoning tools."""

from __future__ import annotations

from agent.integrations.process_docs import ProcessDocumentRetriever
from agent.integrations.process_rag_client import ProcessChangeRAGClient
from agent.integrations.llm_client import SemanticOCRCorrectionClient
from agent.schemas import to_plain_data
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


class ProcessChangeRAGCheckTool:
    name = "Tool_Process_Change_RAG_Check"
    description = "Retrieve dynamic process-change documents and ask an LLM whether release is allowed."

    def run(self, payload: dict, context: ToolContext) -> ToolResult:
        def _call():
            retriever = ProcessDocumentRetriever(context.config)
            retrieved = retriever.retrieve(
                material_id=payload.get("material_id"),
                material=payload.get("material"),
                diameter=payload.get("diameter"),
                workstation=payload.get("workstation", ""),
                task=payload.get("task", ""),
                top_k=context.config.process_rag_top_k,
            )
            retrieved_payload = [to_plain_data(item) for item in retrieved]
            result = ProcessChangeRAGClient(context.config).review(
                task=payload.get("task", ""),
                workstation=payload.get("workstation", ""),
                material_id=payload.get("material_id"),
                material=payload.get("material"),
                diameter=payload.get("diameter"),
                erp_record=payload.get("erp_record") or {},
                retrieved_documents=retrieved_payload,
            )
            data = to_plain_data(result)
            data["retrieved_count"] = len(retrieved_payload)
            return data

        return timed_call(_call)

"""Lightweight retrieval for process-change documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from agent.config import AgentConfig
from agent.schemas import RetrievedDocument


@dataclass
class ProcessDocument:
    document_id: str
    title: str
    path: Path
    text: str


class ProcessDocumentRetriever:
    """Loads local process documents and retrieves relevant snippets.

    This is intentionally lightweight: it supports local txt/md documents now,
    and keeps a clean boundary for replacing the scorer with embeddings later.
    """

    supported_suffixes = {".txt", ".md"}

    def __init__(self, config: AgentConfig):
        self.config = config

    def retrieve(
        self,
        material_id: str | None,
        material: str | None,
        diameter: str | None,
        workstation: str,
        task: str,
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        query_terms = self._query_terms(material_id, material, diameter, workstation, task)
        documents = self._load_documents()
        scored: list[RetrievedDocument] = []
        for doc in documents:
            score = self._score(doc.text, query_terms)
            if score <= 0:
                continue
            scored.append(
                RetrievedDocument(
                    document_id=doc.document_id,
                    title=doc.title,
                    path=str(doc.path),
                    score=score,
                    snippet=self._snippet(doc.text, query_terms),
                    metadata={"source": "local_process_docs"},
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: top_k or self.config.process_rag_top_k]

    def _load_documents(self) -> list[ProcessDocument]:
        docs_dir = self.config.process_rag_docs_dir
        if not docs_dir.exists():
            return []
        documents = []
        for path in sorted(docs_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.supported_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            documents.append(
                ProcessDocument(
                    document_id=path.stem,
                    title=self._title(path, text),
                    path=path,
                    text=text,
                )
            )
        return documents

    def _query_terms(
        self,
        material_id: str | None,
        material: str | None,
        diameter: str | None,
        workstation: str,
        task: str,
    ) -> set[str]:
        seed = " ".join(
            item
            for item in [material_id, material, diameter, workstation, task]
            if item
        )
        terms = set(self._tokens(seed))
        if material_id and "-" in material_id:
            terms.update(part for part in material_id.upper().split("-") if part)
        return terms

    def _score(self, text: str, query_terms: Iterable[str]) -> float:
        normalized = text.upper()
        score = 0.0
        for term in query_terms:
            if not term:
                continue
            count = normalized.count(term.upper())
            if count:
                score += min(count, 5)
        policy_terms = ["复检", "禁止", "暂停", "替换", "临时通知", "工艺变更", "质检", "实验室"]
        score += sum(0.5 for term in policy_terms if term in text)
        return score

    def _snippet(self, text: str, query_terms: Iterable[str], size: int = 700) -> str:
        normalized = text.upper()
        positions = [
            normalized.find(term.upper())
            for term in query_terms
            if term and normalized.find(term.upper()) >= 0
        ]
        start = max(min(positions) - 120, 0) if positions else 0
        snippet = text[start : start + size].replace("\r", " ").strip()
        return snippet

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.upper() for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text)]

    @staticmethod
    def _title(path: Path, text: str) -> str:
        for line in text.splitlines():
            cleaned = line.strip().lstrip("#").strip()
            if cleaned:
                return cleaned[:120]
        return path.stem

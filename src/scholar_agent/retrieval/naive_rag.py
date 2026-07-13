"""Naive RAG baseline: retrieve → stuff context → answer with page citations."""

from __future__ import annotations

from typing import Literal

from scholar_agent.llm.client import ChatMessage, LLMClient
from scholar_agent.llm.prompts import (
    SYSTEM_UNTRUSTED_CONTENT_POLICY,
    build_evidence_prompt_block,
)
from scholar_agent.logging import get_logger
from scholar_agent.models.retrieval import (
    CitationRef,
    NaiveRAGAnswer,
    RetrievalFilters,
    RetrievalHit,
)
from scholar_agent.retrieval.tools import RetrievalToolkit

logger = get_logger(__name__)

SearchMode = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]


class NaiveRAG:
    """Simple retrieve-and-generate baseline (no multi-agent loop)."""

    def __init__(
        self,
        toolkit: RetrievalToolkit,
        *,
        llm: LLMClient | None = None,
        mode: SearchMode | str = "hybrid_rerank",
        top_k: int = 8,
    ) -> None:
        self.toolkit = toolkit
        self.llm = llm
        allowed: set[str] = {"dense", "sparse", "hybrid", "hybrid_rerank"}
        self.mode: SearchMode = mode if mode in allowed else "hybrid_rerank"  # type: ignore[assignment]
        self.top_k = top_k

    def answer(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        use_llm: bool | None = None,
    ) -> NaiveRAGAnswer:
        result = self.toolkit.search(
            query,
            mode=self.mode,
            k=self.top_k,
            filters=filters,
        )
        hits = result.hits[: self.top_k]
        citations = [
            CitationRef(
                paper_id=h.paper_id,
                chunk_id=h.chunk_id,
                page_start=h.page_start,
                page_end=h.page_end,
                marker=f"S{i}",
            )
            for i, h in enumerate(hits, start=1)
        ]

        should_llm = use_llm if use_llm is not None else self.llm is not None
        if should_llm and self.llm is not None and hits:
            answer_text = self._generate_with_llm(query, hits, citations)
            used_llm = True
        else:
            answer_text = self._extractive_answer(query, hits, citations)
            used_llm = False

        # Ensure page references appear
        if hits and not any(c.format_inline() in answer_text for c in citations):
            refs = "; ".join(c.format_inline() for c in citations[:5])
            answer_text = f"{answer_text.rstrip()}\n\nSources: {refs}"

        return NaiveRAGAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            hits=hits,
            method=f"naive_rag:{result.method}",
            used_llm=used_llm,
        )

    def _extractive_answer(
        self,
        query: str,
        hits: list[RetrievalHit],
        citations: list[CitationRef],
    ) -> str:
        if not hits:
            return (
                f"No supporting passages were retrieved for: {query}\n"
                "The corpus may not contain an answer, or indexes may need rebuilding."
            )
        lines = [
            f"Question: {query}",
            "",
            "Evidence-based notes (Naive RAG extractive baseline):",
        ]
        for cit, hit in zip(citations, hits, strict=True):
            snippet = hit.snippet(320)
            lines.append(
                f"- {cit.marker} {cit.format_inline()} "
                f"(chunk={hit.chunk_id}, method={hit.retrieval_method}): {snippet}"
            )
        lines.append("")
        lines.append(
            "These passages are ranked by the configured retrieval stack; "
            "claims above are limited to retrieved text and cite PDF page ranges."
        )
        return "\n".join(lines)

    def _generate_with_llm(
        self,
        query: str,
        hits: list[RetrievalHit],
        citations: list[CitationRef],
    ) -> str:
        assert self.llm is not None
        citation_guide: list[str] = []
        evidence_items: list[tuple[str, str, str]] = []
        for cit, hit in zip(citations, hits, strict=True):
            citation_guide.append(
                f"- {cit.marker}: use {cit.format_inline()} for chunk {hit.chunk_id}"
            )
            evidence_items.append((hit.chunk_id, hit.paper_id, hit.text))
        context = build_evidence_prompt_block(evidence_items)
        citation_mapping = "\n".join(citation_guide)
        system = (
            "You are a literature assistant. Answer ONLY using the provided sources. "
            "Every factual sentence must include an inline citation like "
            "[paper_id p.N] or [paper_id p.N-M] using the page fields given. "
            "If sources are insufficient, say so. Do not invent papers or pages. "
            f"{SYSTEM_UNTRUSTED_CONTENT_POLICY}"
        )
        user = (
            f"Question:\n{query}\n\nAllowed citation mapping:\n"
            f"{citation_mapping}\n\n{context}\n\n"
            "Write a concise answer with inline page citations."
        )
        response = self.llm.chat(
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user),
            ],
            fast=True,
            max_tokens=800,
        )
        return (response.content or "").strip() or self._extractive_answer(query, hits, citations)

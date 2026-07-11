"""Retrieval result types shared across indexes and tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class RetrievalFilters(BaseModel):
    paper_ids: list[str] | None = None
    page_min: int | None = Field(default=None, ge=1)
    page_max: int | None = Field(default=None, ge=1)
    section_contains: str | None = None


class RetrievalHit(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section: str | None = None
    score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    retrieval_method: str

    @field_validator("chunk_id", "paper_id", "text", "retrieval_method")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be non-empty")
        return cleaned

    def snippet(self, max_chars: int = 280) -> str:
        text = " ".join(self.text.split())
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"

    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p.{self.page_start}"
        return f"p.{self.page_start}-{self.page_end}"


class RankTrace(BaseModel):
    """Per-hit debug ranks for dense/sparse/fusion/rerank."""

    chunk_id: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    final_rank: int | None = None


class RetrievalResult(BaseModel):
    query: str
    method: Literal["dense", "sparse", "hybrid", "hybrid_rerank", "graph"]
    hits: list[RetrievalHit] = Field(default_factory=list)
    traces: list[RankTrace] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class CitationRef(BaseModel):
    paper_id: str
    chunk_id: str
    page_start: int
    page_end: int
    marker: str

    def format_inline(self) -> str:
        if self.page_start == self.page_end:
            return f"[{self.paper_id} p.{self.page_start}]"
        return f"[{self.paper_id} p.{self.page_start}-{self.page_end}]"


class NaiveRAGAnswer(BaseModel):
    query: str
    answer: str
    citations: list[CitationRef] = Field(default_factory=list)
    hits: list[RetrievalHit] = Field(default_factory=list)
    method: str = "naive_rag"
    used_llm: bool = False

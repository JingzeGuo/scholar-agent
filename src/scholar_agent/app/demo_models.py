"""Pydantic models for the Streamlit demo and saved-run replay."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from scholar_agent.models.answer import FinalAnswer, SourceCard
from scholar_agent.models.base import ExecutionEvent
from scholar_agent.models.evidence import EvidenceItem
from scholar_agent.models.planning import QueryPlan
from scholar_agent.models.workflow import VerificationResult


class DemoSettings(BaseModel):
    """Interview ablation / mode toggles."""

    compare_naive_rag: bool = False
    enable_graph: bool = True
    enable_corrective: bool = True
    static_routing: bool = False
    verified_evidence_only: bool = True
    embedding_backend: Literal["auto", "hash", "st"] = "hash"
    max_corrective_iterations: int = Field(default=2, ge=0)
    top_k: int = Field(default=8, ge=1)
    use_llm: bool = False
    max_total_tool_calls: int = Field(default=12, ge=1)

    def label(self) -> str:
        parts = [
            "graph" if self.enable_graph else "no-graph",
            "corrective" if self.enable_corrective else "no-corrective",
            "static" if self.static_routing else "adaptive",
            "naive-cmp" if self.compare_naive_rag else "solo",
        ]
        return " · ".join(parts)


class TraceSummary(BaseModel):
    """Compact, UI-friendly research trace."""

    query_type: str | None = None
    answer_type: str | None = None
    sub_questions: list[dict[str, Any]] = Field(default_factory=list)
    tool_events: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_methods: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    verified_evidence_count: int = 0
    coverage_score: float | None = None
    is_sufficient: bool | None = None
    corrective_iterations: int = 0
    corrective_queries: list[str] = Field(default_factory=list)
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    citation_valid: bool | None = None
    citation_issue_count: int = 0
    terminated_reason: str | None = None
    unanswerable: bool = False
    latency_ms: int = 0
    tool_call_count: int = 0
    token_estimate: int = 0


class NaiveComparisonView(BaseModel):
    answer: str = ""
    method: str = "naive_rag"
    citations: list[dict[str, Any]] = Field(default_factory=list)
    hit_count: int = 0
    latency_ms: int = 0
    used_llm: bool = False


class DemoSessionResult(BaseModel):
    """Live or replayed demo session payload for the UI."""

    run_id: str
    query: str
    settings: DemoSettings
    offline_replay: bool = False
    answer_markdown: str = ""
    claims: list[dict[str, Any]] = Field(default_factory=list)
    source_cards: list[SourceCard] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    plan: QueryPlan | None = None
    verification: VerificationResult | None = None
    final_answer: FinalAnswer | None = None
    events: list[ExecutionEvent] = Field(default_factory=list)
    trace: TraceSummary = Field(default_factory=TraceSummary)
    naive: NaiveComparisonView | None = None
    status: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class SavedDemoRun(BaseModel):
    """Precomputed interview demo that works without live indexes/API."""

    demo_id: str
    title: str
    query: str
    settings: DemoSettings
    created_at: str
    offline: bool = True
    notes: str = ""
    session: DemoSessionResult

    @field_validator("demo_id", "title", "query")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be non-empty")
        return cleaned

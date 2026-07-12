"""Research workflow state and verification result models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from scholar_agent.models.answer import CitationReport, DraftAnswer, FinalAnswer
from scholar_agent.models.base import BudgetStatus, ExecutionEvent, TokenUsage
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.models.planning import QueryPlan


class CorrectiveQuery(BaseModel):
    """Actionable retrieval request tied to the plan gap it must repair."""

    query: str
    target_sub_question_id: str
    missing_aspect: str

    @field_validator("query", "target_sub_question_id", "missing_aspect")
    @classmethod
    def _corrective_fields_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be non-empty")
        return cleaned


class VerificationResult(BaseModel):
    is_sufficient: bool
    coverage_score: float = Field(ge=0.0, le=1.0)
    covered_sub_questions: list[str] = Field(default_factory=list)
    supported_evidence_ids: dict[str, list[str]] = Field(default_factory=dict)
    missing_sub_questions: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    corrective_queries: list[str] = Field(default_factory=list)
    corrective_actions: list[CorrectiveQuery] = Field(default_factory=list)
    unanswerable: bool = False
    # Concise decision explanation — never private chain-of-thought
    rationale_summary: str

    @field_validator("rationale_summary")
    @classmethod
    def _summary_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("rationale_summary must be non-empty")
        return cleaned


class ResearchRunState(BaseModel):
    """Serializable snapshot of a full research run.

    Used for persistence, debugging, and demo precomputation. LangGraph runtime
    state may mirror these fields with reducers in ``agents.state``.
    """

    run_id: str
    query: str
    plan: QueryPlan | None = None
    active_sub_questions: list[str] = Field(default_factory=list)
    evidence_ledger: EvidenceLedger = Field(default_factory=EvidenceLedger)
    verification: VerificationResult | None = None
    corrective_queries: list[str] = Field(default_factory=list)
    iteration: int = 0
    tool_call_count: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0
    budgets: BudgetStatus = Field(default_factory=BudgetStatus)
    execution_events: list[ExecutionEvent] = Field(default_factory=list)
    draft_answer: DraftAnswer | None = None
    final_answer: FinalAnswer | None = None
    citation_report: CitationReport | None = None
    errors: list[str] = Field(default_factory=list)

    @field_validator("run_id", "query")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    def evidence_items(self) -> list[EvidenceItem]:
        return list(self.evidence_ledger.items)

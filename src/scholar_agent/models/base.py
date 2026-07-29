"""Runtime events, budgets, and shared compatibility models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class QueryType(StrEnum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    COMPARISON = "comparison"
    RELATIONAL = "relational"
    SYNTHESIS = "synthesis"


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    PLAN_CREATED = "plan_created"
    TOOL_SELECTED = "tool_selected"
    TOOL_RESULT = "tool_result"
    EVIDENCE_ADDED = "evidence_added"
    VERIFICATION = "verification"
    CORRECTIVE = "corrective"
    ANSWER_DRAFTED = "answer_drafted"
    CITATION_VALIDATED = "citation_validated"
    BUDGET_HIT = "budget_hit"
    ERROR = "error"
    # Agent-loop lifecycle events
    ITERATION = "iteration"
    DECISION = "decision"
    TERMINATED = "terminated"


class ErrorCategory(StrEnum):
    """Coarse classification for structured error events."""

    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    PROVIDER = "provider"
    INDEX_UNAVAILABLE = "index_unavailable"
    CACHE = "cache"
    BUDGET = "budget"
    TOOL = "tool"
    UNKNOWN = "unknown"


class StructuredError(BaseModel):
    """Sanitized, structured error for logs and execution events.

    Never include secrets, full environment dumps, auth headers, or private CoT.
    """

    run_id: str | None = None
    component: str
    operation: str
    category: ErrorCategory = ErrorCategory.UNKNOWN
    retryable: bool = False
    message: str
    fallback_used: str | None = None
    timestamp: str = Field(default_factory=utc_now_iso)
    duration_ms: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _message_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message must be non-empty")
        # Hard cap to avoid dumping huge stack traces into events
        return cleaned[:500]

    def to_event_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class ExecutionEvent(BaseModel):
    """Auditable agent event. Never stores private chain-of-thought."""

    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex[:12]}")
    run_id: str
    event_type: EventType
    timestamp: str = Field(default_factory=utc_now_iso)
    component: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("summary")
    @classmethod
    def _summary_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("summary must be non-empty")
        return cleaned


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class BudgetStatus(BaseModel):
    tool_call_count: int = 0
    max_tool_calls: int = 4
    iteration: int = 0
    max_iterations: int = 3
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    max_total_tokens: int = 100_000
    latency_ms: int = 0
    max_latency_ms: int = 120_000
    terminated_reason: str | None = None

    def tool_budget_remaining(self) -> int:
        return max(0, self.max_tool_calls - self.tool_call_count)

    def iteration_budget_remaining(self) -> int:
        return max(0, self.max_iterations - self.iteration)

    def is_exhausted(self) -> bool:
        return (
            self.tool_call_count >= self.max_tool_calls
            or self.iteration >= self.max_iterations
            or self.token_usage.total_tokens >= self.max_total_tokens
            or self.latency_ms >= self.max_latency_ms
        )


class CompatibilityDecision(BaseModel):
    """Small structured-output schema used by provider compatibility checks."""

    action: Literal["retrieve", "verify", "finish"]
    reason: str
    need_more_evidence: bool = False

"""Query routing / retrieval policy models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from scholar_agent.models.base import QueryType


class RetrievalPolicy(StrEnum):
    """Concrete tool/policy the Research Agent can execute."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"
    GRAPH = "graph"
    HYBRID_PLUS_GRAPH = "hybrid_plus_graph"


class RoutingDecision(BaseModel):
    """Router recommendation for a (sub-)question."""

    query: str
    query_type: QueryType
    recommended_policy: RetrievalPolicy
    rationale: str
    signals: list[str] = Field(default_factory=list)
    # Agent may override within budget; both are logged
    allow_override: bool = True

    @field_validator("query", "rationale")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be non-empty")
        return cleaned


class ToolAction(BaseModel):
    """One Research Agent tool decision (auditable, not chain-of-thought)."""

    tool_name: str
    policy: RetrievalPolicy
    query: str
    recommended_policy: RetrievalPolicy
    overridden: bool = False
    reason: str

    @field_validator("tool_name", "query", "reason")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be non-empty")
        return cleaned

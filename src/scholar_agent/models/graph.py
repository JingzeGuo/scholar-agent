"""Knowledge-graph entity and relation models (evidence-linked)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class EntityType(StrEnum):
    PAPER = "Paper"
    METHOD = "Method"
    DATASET = "Dataset"
    TASK = "Task"
    METRIC = "Metric"
    AUTHOR = "Author"
    ORGANIZATION = "Organization"


class RelationType(StrEnum):
    PROPOSES = "PROPOSES"
    EXTENDS = "EXTENDS"
    USES = "USES"
    EVALUATES_ON = "EVALUATES_ON"
    REPORTS = "REPORTS"
    COMPARES_WITH = "COMPARES_WITH"
    OUTPERFORMS = "OUTPERFORMS"
    CITES = "CITES"
    AUTHORED_BY = "AUTHORED_BY"


class Entity(BaseModel):
    entity_id: str
    entity_type: EntityType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None

    @field_validator("entity_id", "canonical_name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @field_validator("aliases")
    @classmethod
    def _clean_aliases(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for alias in values:
            cleaned = alias.strip()
            if cleaned and cleaned.lower() not in seen:
                out.append(cleaned)
                seen.add(cleaned.lower())
        return out


class Relation(BaseModel):
    """Schema-constrained relation that always carries source evidence."""

    relation_id: str
    subject_surface: str
    object_surface: str
    subject_entity_id: str | None = None
    object_entity_id: str | None = None
    subject_type: EntityType | None = None
    object_type: EntityType | None = None
    relation_type: RelationType
    evidence_span: str
    paper_id: str
    chunk_id: str
    page_number: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator(
        "relation_id",
        "subject_surface",
        "object_surface",
        "evidence_span",
        "paper_id",
        "chunk_id",
    )
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @model_validator(mode="after")
    def _require_evidence_span(self) -> Relation:
        if not self.evidence_span.strip():
            raise ValueError("relations without an evidence span are discarded")
        return self

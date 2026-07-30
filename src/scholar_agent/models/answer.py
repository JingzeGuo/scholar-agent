"""Writer and citation validation models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class AnswerStatus(StrEnum):
    """Truthful answer state after evidence verification."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class ClaimWithCitations(BaseModel):
    claim_id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    sub_question_id: str | None = None
    requirement_key: str | None = None
    entity_id: str | None = None
    dimension: str | None = None

    @field_validator(
        "claim_id",
        "text",
        "sub_question_id",
        "requirement_key",
        "entity_id",
        "dimension",
    )
    @classmethod
    def _non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned


class ComparisonCell(BaseModel):
    """One entity's evidence-backed answer for a comparison dimension."""

    entity_id: str
    entity_label: str
    text: str = "Insufficient verified evidence"
    evidence_ids: list[str] = Field(default_factory=list)
    claim_id: str | None = None
    supported: bool = False

    @field_validator("entity_id", "entity_label", "text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @field_validator("claim_id")
    @classmethod
    def _optional_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @model_validator(mode="after")
    def _supported_cell_has_provenance(self) -> ComparisonCell:
        if self.supported and (not self.claim_id or not self.evidence_ids):
            raise ValueError("supported comparison cells require claim_id and evidence_ids")
        return self


class ComparisonRow(BaseModel):
    """A stable comparison dimension with one cell per target entity."""

    requirement_key: str
    dimension: str
    label: str
    cells: list[ComparisonCell] = Field(default_factory=list)

    @field_validator("requirement_key", "dimension", "label")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @field_validator("cells")
    @classmethod
    def _unique_entities(cls, value: list[ComparisonCell]) -> list[ComparisonCell]:
        entity_ids = [cell.entity_id for cell in value]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("comparison row entity_ids must be unique")
        return value


class SourceCard(BaseModel):
    """Machine-readable provenance card for a cited evidence item."""

    evidence_id: str
    paper_id: str
    chunk_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    snippet: str = ""
    retrieval_method: str | None = None
    title: str | None = None
    pdf_path: str | None = None

    @field_validator("evidence_id", "paper_id", "chunk_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p.{self.page_start}"
        return f"p.{self.page_start}-{self.page_end}"

    def format_inline(self) -> str:
        return f"[{self.paper_id} {self.page_label()}]"

    def format_reference(self) -> str:
        label = self.title or self.paper_id
        reference = (
            f"{label} [{self.paper_id}] {self.page_label()} "
            f"(chunk={self.chunk_id}, evidence={self.evidence_id})"
        )
        if self.pdf_path:
            return f"{reference} · PDF={self.pdf_path}"
        return reference


class DraftAnswer(BaseModel):
    claims: list[ClaimWithCitations] = Field(default_factory=list)
    markdown: str = ""
    core_answer: str = ""
    rows: list[ComparisonRow] = Field(default_factory=list)
    status: AnswerStatus = AnswerStatus.INSUFFICIENT
    corpus_insufficient: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_status(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "status" in data:
            return data
        migrated = dict(data)
        claims = migrated.get("claims") or []
        if migrated.get("corpus_insufficient"):
            migrated["status"] = (
                AnswerStatus.PARTIAL if claims else AnswerStatus.INSUFFICIENT
            )
        else:
            migrated["status"] = (
                AnswerStatus.COMPLETE if claims else AnswerStatus.INSUFFICIENT
            )
        return migrated

    @model_validator(mode="after")
    def _derive_legacy_insufficiency(self) -> DraftAnswer:
        self.corpus_insufficient = self.status != AnswerStatus.COMPLETE
        return self


class CitationIssue(BaseModel):
    severity: str
    claim_id: str | None = None
    evidence_id: str | None = None
    message: str

    @field_validator("severity", "message")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned


class CitationReport(BaseModel):
    is_valid: bool
    issues: list[CitationIssue] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    cited_paper_ids: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    markdown: str
    claims: list[ClaimWithCitations] = Field(default_factory=list)
    core_answer: str = ""
    rows: list[ComparisonRow] = Field(default_factory=list)
    status: AnswerStatus = AnswerStatus.INSUFFICIENT
    sources: list[str] = Field(default_factory=list)
    source_cards: list[SourceCard] = Field(default_factory=list)
    citation_report: CitationReport | None = None
    corpus_insufficient: bool = False

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_status(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "status" in data:
            return data
        migrated = dict(data)
        claims = migrated.get("claims") or []
        if migrated.get("corpus_insufficient"):
            migrated["status"] = (
                AnswerStatus.PARTIAL if claims else AnswerStatus.INSUFFICIENT
            )
        else:
            migrated["status"] = (
                AnswerStatus.COMPLETE if claims else AnswerStatus.INSUFFICIENT
            )
        return migrated

    @model_validator(mode="after")
    def _derive_legacy_insufficiency(self) -> FinalAnswer:
        self.corpus_insufficient = self.status != AnswerStatus.COMPLETE
        return self

"""Writer and citation validation models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ClaimWithCitations(BaseModel):
    claim_id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("claim_id", "text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned


class DraftAnswer(BaseModel):
    claims: list[ClaimWithCitations] = Field(default_factory=list)
    markdown: str = ""
    corpus_insufficient: bool = False
    notes: list[str] = Field(default_factory=list)


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
    sources: list[str] = Field(default_factory=list)
    citation_report: CitationReport | None = None
    corpus_insufficient: bool = False

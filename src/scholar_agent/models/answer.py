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
    source_cards: list[SourceCard] = Field(default_factory=list)
    citation_report: CitationReport | None = None
    corpus_insufficient: bool = False

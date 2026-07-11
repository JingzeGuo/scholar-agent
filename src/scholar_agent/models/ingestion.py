"""Ingestion quality reports and intermediate extraction structures."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class ExtractionSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ExtractionIssue(BaseModel):
    code: str
    severity: ExtractionSeverity
    message: str
    page_number: int | None = None

    @field_validator("code", "message")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be non-empty")
        return cleaned


class SectionBlock(BaseModel):
    """Contiguous text under one section heading (may span pages)."""

    title: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str

    @field_validator("text")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("section text must be non-empty")
        return value


class PaperExtractionReport(BaseModel):
    paper_id: str
    pdf_path: str
    page_count: int = Field(ge=0)
    empty_page_count: int = Field(ge=0)
    scanned_suspect_page_count: int = Field(ge=0)
    total_chars: int = Field(ge=0)
    total_tokens_est: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    is_empty_paper: bool = False
    is_scanned_suspect: bool = False
    skipped: bool = False
    skip_reason: str | None = None
    issues: list[ExtractionIssue] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == ExtractionSeverity.ERROR for i in self.issues)


class CorpusIngestionReport(BaseModel):
    """Aggregate report written after an ingest run."""

    run_id: str
    manifest_path: str
    processed_dir: str
    papers_attempted: int = 0
    papers_ingested: int = 0
    papers_skipped: int = 0
    papers_failed: int = 0
    total_pages: int = 0
    total_chunks: int = 0
    empty_papers: list[str] = Field(default_factory=list)
    scanned_suspect_papers: list[str] = Field(default_factory=list)
    paper_reports: list[PaperExtractionReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

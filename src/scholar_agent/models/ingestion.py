"""Ingestion quality reports and intermediate extraction structures."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


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


class SectionPageText(BaseModel):
    """Text contributed by one PDF page to a section.

    ``SectionBlock.text`` is retained for backwards compatibility.  New
    ingestion runs also populate this page-level representation so token
    windows can carry exact PDF provenance instead of estimating pages from a
    section-wide token ratio.
    """

    page_number: int = Field(ge=1)
    text: str

    @field_validator("text")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("page text must be non-empty")
        return cleaned


class SectionBlock(BaseModel):
    """Contiguous text under one section heading (may span pages)."""

    title: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str
    page_texts: list[SectionPageText] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("section text must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate_page_texts(self) -> SectionBlock:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        if not self.page_texts:
            return self
        page_numbers = [item.page_number for item in self.page_texts]
        if page_numbers != sorted(set(page_numbers)):
            raise ValueError("page_texts must have unique, increasing page numbers")
        if page_numbers[0] < self.page_start or page_numbers[-1] > self.page_end:
            raise ValueError("page_texts must fall within the section page range")
        return self


class PaperExtractionReport(BaseModel):
    paper_id: str
    pdf_path: str
    page_count: int = Field(ge=0)
    empty_page_count: int = Field(ge=0)
    scanned_suspect_page_count: int = Field(ge=0)
    total_chars: int = Field(ge=0)
    total_tokens_est: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    tokenizer_encoding: str | None = None
    tokenizer_backend: str | None = None
    ingestion_config_fingerprint: str | None = None
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
    tokenizer_encoding: str
    tokenizer_backend: str
    ingestion_config_fingerprint: str
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

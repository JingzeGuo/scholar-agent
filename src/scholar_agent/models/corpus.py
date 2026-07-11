"""Paper, page, chunk, and corpus manifest models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class IngestionStatus(StrEnum):
    PENDING = "pending"
    INGESTED = "ingested"
    FAILED = "failed"
    SKIPPED = "skipped"


class CorpusManifestEntry(BaseModel):
    """One row in ``data/corpus_manifest.jsonl``."""

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_filename: str
    source_url: str | None = None
    topic_labels: list[str] = Field(default_factory=list)
    ingestion_status: IngestionStatus = IngestionStatus.PENDING
    content_hash: str

    @field_validator("paper_id", "title", "pdf_filename", "content_hash")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @field_validator("year")
    @classmethod
    def _reasonable_year(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value < 1900 or value > 2100:
            raise ValueError(f"year out of range: {value}")
        return value

    @field_validator("authors", "topic_labels")
    @classmethod
    def _strip_list(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if v and v.strip()]

    @model_validator(mode="after")
    def _require_identity_signal(self) -> CorpusManifestEntry:
        if not self.doi and not self.arxiv_id and not self.title:
            raise ValueError("manifest entry needs doi, arxiv_id, or title")
        return self


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    source_url: str | None = None
    pdf_path: str
    content_hash: str
    topic_labels: list[str] = Field(default_factory=list)
    page_count: int | None = None

    @field_validator("paper_id", "title", "pdf_path", "content_hash")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @field_validator("page_count")
    @classmethod
    def _page_count_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("page_count must be >= 0")
        return value

    def pdf_path_obj(self) -> Path:
        return Path(self.pdf_path)


class PaperPage(BaseModel):
    """Single PDF page of extracted text (preserves page provenance)."""

    paper_id: str
    page_number: int = Field(ge=1)
    text: str
    char_count: int = Field(ge=0)
    is_empty: bool = False
    is_scanned_suspect: bool = False

    @model_validator(mode="after")
    def _sync_empty_flag(self) -> PaperPage:
        if not self.text.strip():
            object.__setattr__(self, "is_empty", True)
        return self


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section: str | None = None
    token_count: int = Field(ge=0)
    content_hash: str

    @field_validator("chunk_id", "paper_id", "text", "content_hash")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @model_validator(mode="after")
    def _validate_page_range(self) -> Chunk:
        if self.page_end < self.page_start:
            raise ValueError(
                f"page_end ({self.page_end}) must be >= page_start ({self.page_start})"
            )
        return self

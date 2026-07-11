"""Deterministic, stable identifiers for corpus and runtime objects.

IDs are content-addressed where possible so re-running ingestion or evidence
construction yields the same identifiers. Run IDs remain random (UUID-based)
because each research execution is a distinct audit trail.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from uuid import uuid4

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(value: str) -> str:
    """Normalize Unicode, case, punctuation, and whitespace for hashing / aliases."""
    text = unicodedata.normalize("NFKC", value).strip().lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def normalize_for_id(value: str) -> str:
    """Stronger normalization for ID components (alnum + single underscores)."""
    text = normalize_text(value)
    text = _NON_ALNUM_RE.sub("_", text).strip("_")
    return text or "empty"


def content_hash(value: str | bytes, *, length: int = 16) -> str:
    """Return a truncated SHA-256 hex digest of text or bytes."""
    if length < 8 or length > 64:
        raise ValueError("length must be between 8 and 64")
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()[:length]


def _compose(*parts: str, prefix: str, length: int = 16) -> str:
    material = "|".join(parts)
    return f"{prefix}_{content_hash(material, length=length)}"


def new_run_id() -> str:
    """Generate a unique run identifier (not content-addressed)."""
    return f"run_{uuid4().hex[:16]}"


def make_paper_id(
    *,
    arxiv_id: str | None = None,
    doi: str | None = None,
    title: str | None = None,
    year: int | None = None,
    content: str | None = None,
) -> str:
    """Derive a stable paper ID.

    Preference order:
    1. arXiv ID (normalized)
    2. DOI (normalized)
    3. title + year (+ optional content hash fragment)
    """
    if arxiv_id and arxiv_id.strip():
        cleaned = normalize_for_id(arxiv_id.replace("/", "_"))
        return f"paper_arxiv_{cleaned}"
    if doi and doi.strip():
        cleaned = normalize_for_id(doi)
        return f"paper_doi_{cleaned}"
    if title and title.strip():
        year_part = str(year) if year is not None else "na"
        base = _compose(
            normalize_text(title),
            year_part,
            content_hash(content) if content else "no_content",
            prefix="paper",
        )
        return base
    raise ValueError("make_paper_id requires arxiv_id, doi, or title")


def make_chunk_id(
    paper_id: str,
    *,
    page_start: int,
    page_end: int,
    text: str,
    section: str | None = None,
) -> str:
    """Stable chunk ID from paper, pages, optional section, and text hash."""
    if page_start < 1 or page_end < page_start:
        raise ValueError(f"invalid page range: {page_start}-{page_end}")
    return _compose(
        paper_id,
        str(page_start),
        str(page_end),
        normalize_text(section or ""),
        content_hash(text),
        prefix="chunk",
    )


def make_entity_id(entity_type: str, canonical_name: str) -> str:
    """Stable entity ID from type and canonical name."""
    if not entity_type.strip():
        raise ValueError("entity_type must be non-empty")
    if not canonical_name.strip():
        raise ValueError("canonical_name must be non-empty")
    return _compose(
        normalize_for_id(entity_type),
        normalize_text(canonical_name),
        prefix="ent",
    )


def make_relation_id(
    *,
    subject_entity_id: str,
    relation_type: str,
    object_entity_id: str,
    chunk_id: str,
    evidence_span: str,
) -> str:
    """Stable relation ID including provenance chunk and evidence span."""
    return _compose(
        subject_entity_id,
        normalize_for_id(relation_type),
        object_entity_id,
        chunk_id,
        content_hash(normalize_text(evidence_span)),
        prefix="rel",
    )


def make_evidence_id(
    *,
    run_id: str,
    chunk_id: str,
    evidence_text: str,
    sub_question_id: str | None = None,
) -> str:
    """Stable within-run evidence ID from chunk + normalized span (+ sub-question)."""
    return _compose(
        run_id,
        chunk_id,
        content_hash(normalize_text(evidence_text)),
        sub_question_id or "",
        prefix="ev",
    )


def make_sub_question_id(plan_seed: str, question: str, index: int) -> str:
    """Stable sub-question ID within a plan."""
    return _compose(plan_seed, str(index), normalize_text(question), prefix="sq", length=12)

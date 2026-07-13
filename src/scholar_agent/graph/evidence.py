"""Evidence-span validation: relations must quote source chunk text."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from scholar_agent.ids import normalize_text
from scholar_agent.models.corpus import Chunk, PaperPage
from scholar_agent.models.graph import Relation

_WS = re.compile(r"\s+")


def normalize_span(text: str) -> str:
    """Normalize for soft substring matching (case/WS/unicode)."""
    value = unicodedata.normalize("NFKC", text).lower()
    value = _WS.sub(" ", value).strip()
    return value


def find_evidence_span(chunk_text: str, span: str) -> str | None:
    """Return a concrete evidence span from chunk_text, or None if unsupported.

    Prefers exact substring match; falls back to whitespace-normalized match
    and returns the original chunk slice when possible.
    """
    span = span.strip()
    if not span:
        return None
    if span in chunk_text:
        return span

    # Case-insensitive exact
    lower_chunk = chunk_text.lower()
    lower_span = span.lower()
    idx = lower_chunk.find(lower_span)
    if idx >= 0:
        return chunk_text[idx : idx + len(span)]

    # Normalized whitespace match
    norm_chunk = normalize_span(chunk_text)
    norm_span = normalize_span(span)
    if not norm_span or norm_span not in norm_chunk:
        # Try if span is a significant substring of chunk after normalizing
        if len(norm_span) >= 12 and any(
            normalize_text(norm_span) in normalize_text(sent)
            for sent in re.split(r"[.!?]\s+", chunk_text)
        ):
            return span
        return None

    # Approximate: return the provided span (already validated as norm-present)
    return span


def validate_relation_against_chunk(relation: Relation, chunk: Chunk) -> Relation | None:
    """Keep relation only if evidence_span is grounded in the chunk text."""
    if relation.chunk_id != chunk.chunk_id:
        return None
    grounded = find_evidence_span(chunk.text, relation.evidence_span)
    if grounded is None:
        return None
    if relation.paper_id != chunk.paper_id:
        return None
    # Keep a conservative range until the graph pipeline can localize the span
    # against physical PDF pages.  Old serialized relations only have
    # ``page_number``; in that case a multi-page chunk must not be represented
    # as if the evidence were proven to occur on its first page.
    page_start = relation.page_number
    page_end = relation.page_end or page_start
    if (
        page_start < chunk.page_start
        or page_start > chunk.page_end
        or page_end < page_start
        or page_end > chunk.page_end
    ):
        page_start, page_end = chunk.page_start, chunk.page_end
    return relation.model_copy(
        update={
            "evidence_span": grounded,
            "page_number": page_start,
            "page_end": page_end,
        }
    )


def locate_evidence_pages(
    evidence_span: str,
    pages: Sequence[PaperPage],
    *,
    page_start: int,
    page_end: int,
) -> tuple[int, int] | None:
    """Locate a grounded span on one or more physical PDF pages.

    Ingestion removes repeated headers/footers and may join a sentence across
    page boundaries.  Searching both individual pages and ordered contiguous
    page windows preserves those cases while rejecting the old shortcut of
    assigning every relation to ``chunk.page_start``.
    """
    normalized_span = normalize_span(evidence_span)
    if not normalized_span:
        return None
    candidates = sorted(
        (page for page in pages if page_start <= page.page_number <= page_end),
        key=lambda page: page.page_number,
    )
    if not candidates:
        return None

    normalized_pages = [normalize_span(page.text) for page in candidates]
    for page, text in zip(candidates, normalized_pages, strict=True):
        if normalized_span in text:
            return page.page_number, page.page_number

    # A chunk range is normally one or two pages, but the generic contiguous
    # search also covers longer reference sections.  Do not skip page numbers:
    # evidence cannot jump over an unrelated physical page.
    # Search shortest windows first.  Otherwise a span on pages 2–3 of a
    # three-page chunk would be reported conservatively but imprecisely as 1–3.
    for width in range(2, len(candidates) + 1):
        for start_index in range(0, len(candidates) - width + 1):
            window = candidates[start_index : start_index + width]
            if any(
                right.page_number != left.page_number + 1
                for left, right in zip(window, window[1:], strict=False)
            ):
                continue
            combined = " ".join(normalized_pages[start_index : start_index + width])
            if normalized_span in combined:
                return window[0].page_number, window[-1].page_number
    return None


def localize_relation_to_pages(
    relation: Relation,
    chunk: Chunk,
    pages: Sequence[PaperPage],
) -> Relation | None:
    """Return a relation with exact physical page provenance, or ``None``."""
    grounded = validate_relation_against_chunk(relation, chunk)
    if grounded is None:
        return None
    location = locate_evidence_pages(
        grounded.evidence_span,
        pages,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
    )
    if location is None:
        return None
    start, end = location
    return grounded.model_copy(update={"page_number": start, "page_end": end})


def validate_relations(
    relations: list[Relation],
    chunks_by_id: dict[str, Chunk],
) -> list[Relation]:
    """Filter relations that lack grounded evidence spans."""
    kept: list[Relation] = []
    for rel in relations:
        chunk = chunks_by_id.get(rel.chunk_id)
        if chunk is None:
            continue
        ok = validate_relation_against_chunk(rel, chunk)
        if ok is not None:
            kept.append(ok)
    return kept

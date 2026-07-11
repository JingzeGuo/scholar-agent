"""Evidence-span validation: relations must quote source chunk text."""

from __future__ import annotations

import re
import unicodedata

from scholar_agent.ids import normalize_text
from scholar_agent.models.corpus import Chunk
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
    # Align page with chunk provenance when possible
    page = relation.page_number
    if page < chunk.page_start or page > chunk.page_end:
        page = chunk.page_start
    return relation.model_copy(update={"evidence_span": grounded, "page_number": page})


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

"""Section-aware, token-aware chunking with stable IDs and page provenance."""

from __future__ import annotations

from scholar_agent.ids import content_hash, make_chunk_id
from scholar_agent.ingestion.tokens import count_tokens, decode_tokens, encode_tokens
from scholar_agent.models.corpus import Chunk
from scholar_agent.models.ingestion import SectionBlock


def chunk_sections(
    paper_id: str,
    sections: list[SectionBlock],
    *,
    target_tokens: int = 600,
    overlap_tokens: int = 80,
    min_tokens: int = 80,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Chunk section blocks without crossing paper boundaries.

    Prefers keeping sections intact when they fit under ``target_tokens``;
    otherwise splits within a section using token windows with overlap.
    """
    if target_tokens < 50:
        raise ValueError("target_tokens must be >= 50")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be < target_tokens")
    if min_tokens > target_tokens:
        raise ValueError("min_tokens must be <= target_tokens")

    chunks: list[Chunk] = []
    for section in sections:
        section_tokens = count_tokens(section.text, encoding_name=encoding_name)
        if section_tokens == 0:
            continue
        if section_tokens <= target_tokens:
            chunks.append(
                _make_chunk(
                    paper_id=paper_id,
                    text=section.text,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    section=section.title,
                    encoding_name=encoding_name,
                )
            )
            continue
        chunks.extend(
            _split_long_section(
                paper_id=paper_id,
                section=section,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                min_tokens=min_tokens,
                encoding_name=encoding_name,
            )
        )
    return chunks


def _make_chunk(
    *,
    paper_id: str,
    text: str,
    page_start: int,
    page_end: int,
    section: str | None,
    encoding_name: str,
) -> Chunk:
    cleaned = text.strip()
    token_count = count_tokens(cleaned, encoding_name=encoding_name)
    return Chunk(
        chunk_id=make_chunk_id(
            paper_id,
            page_start=page_start,
            page_end=page_end,
            text=cleaned,
            section=section,
        ),
        paper_id=paper_id,
        text=cleaned,
        page_start=page_start,
        page_end=page_end,
        section=section,
        token_count=token_count,
        content_hash=content_hash(cleaned),
    )


def _split_long_section(
    *,
    paper_id: str,
    section: SectionBlock,
    target_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    encoding_name: str,
) -> list[Chunk]:
    """Split a long section by tokens with exact page provenance when available."""
    token_ids, token_pages = _section_token_stream(section, encoding_name=encoding_name)
    if not token_ids:
        return []

    chunks: list[Chunk] = []
    start = 0
    n = len(token_ids)
    step = max(1, target_tokens - overlap_tokens)

    while start < n:
        end = min(start + target_tokens, n)
        window = token_ids[start:end]
        # Avoid tiny trailing fragments when possible by merging backward
        if end == n and len(window) < min_tokens and chunks:
            # extend previous chunk conceptually by re-encoding join — simpler: emit if non-empty
            pass
        text = decode_tokens(window, encoding_name=encoding_name).strip()
        if text:
            if token_pages is None:
                # Backwards-compatible SectionBlock values do not contain a
                # page→text map.  The complete section range is conservative
                # and truthful; a proportional guess is not.
                page_start, page_end = section.page_start, section.page_end
            else:
                # Inter-page separators carry formatting but no source claim.
                # Ignoring their sentinel prevents a window that ends exactly
                # on ``\n\n`` from spuriously including the following page.
                window_pages = [page for page in token_pages[start:end] if page is not None]
                if not window_pages:
                    # A stripped, non-empty decoded window should always own
                    # at least one content token.  Keep this guard explicit so
                    # future tokenizer changes cannot fabricate provenance.
                    raise ValueError("non-empty chunk window has no source page")
                page_start, page_end = min(window_pages), max(window_pages)
            chunks.append(
                _make_chunk(
                    paper_id=paper_id,
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    section=section.title,
                    encoding_name=encoding_name,
                )
            )
        if end >= n:
            break
        start += step

    # Drop a trailing micro-chunk by merging into previous when very small
    if len(chunks) >= 2 and chunks[-1].token_count < min_tokens:
        prev, last = chunks[-2], chunks[-1]
        merged_text = (prev.text + "\n\n" + last.text).strip()
        chunks[-2] = _make_chunk(
            paper_id=paper_id,
            text=merged_text,
            page_start=prev.page_start,
            page_end=last.page_end,
            section=section.title,
            encoding_name=encoding_name,
        )
        chunks.pop()
    return chunks


def _section_token_stream(
    section: SectionBlock,
    *,
    encoding_name: str,
) -> tuple[list[int], list[int | None] | None]:
    """Return token IDs and their originating PDF page numbers.

    Separators use a ``None`` sentinel because they carry no source claim.
    Attribution to either neighboring page would make a token window ending or
    beginning exactly on the separator report a page with no chunk content.
    """
    if not section.page_texts:
        return encode_tokens(section.text, encoding_name=encoding_name), None

    token_ids: list[int] = []
    token_pages: list[int | None] = []
    for item in section.page_texts:
        if token_ids:
            separator = encode_tokens("\n\n", encoding_name=encoding_name)
            token_ids.extend(separator)
            token_pages.extend([None] * len(separator))
        page_tokens = encode_tokens(item.text, encoding_name=encoding_name)
        token_ids.extend(page_tokens)
        token_pages.extend([item.page_number] * len(page_tokens))
    return token_ids, token_pages

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
    """Split a long section by tokens; approximate page range from section span."""
    token_ids = encode_tokens(section.text, encoding_name=encoding_name)
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
            # Page range: proportional estimate within section pages
            page_start, page_end = _estimate_pages(
                section, start=start, end=end, total_tokens=n
            )
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


def _estimate_pages(
    section: SectionBlock,
    *,
    start: int,
    end: int,
    total_tokens: int,
) -> tuple[int, int]:
    if section.page_start == section.page_end or total_tokens <= 0:
        return section.page_start, section.page_end
    span = section.page_end - section.page_start
    frac_a = start / total_tokens
    frac_b = max(start + 1, end) / total_tokens
    page_a = section.page_start + int(span * frac_a)
    page_b = section.page_start + int(span * frac_b)
    page_a = min(max(page_a, section.page_start), section.page_end)
    page_b = min(max(page_b, page_a), section.page_end)
    return page_a, page_b

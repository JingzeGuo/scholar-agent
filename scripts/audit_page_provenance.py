#!/usr/bin/env python3
"""Read-only audit of canonical chunk-to-PDF page provenance.

The audit re-extracts each source PDF with the same loader and repeated
header/footer cleanup used by ingestion.  It then checks that both edges of a
canonical chunk have a stable content anchor on the chunk's declared boundary
pages.  Requiring both edges is important: merely finding a chunk somewhere
inside a declared range would incorrectly accept an over-wide range.

The command never modifies corpus data.  It prints either a concise summary or
the complete, Pydantic-validated JSON report and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from scholar_agent.ids import content_hash
from scholar_agent.ingestion.headers import strip_headers_footers
from scholar_agent.ingestion.loader import PDFLoadError, file_content_hash, load_pages
from scholar_agent.ingestion.sections import pages_to_sections
from scholar_agent.models.corpus import Chunk, Paper, PaperPage
from scholar_agent.retrieval.chunk_store import corpus_fingerprint
from scholar_agent.storage.jsonl import JsonlRepository

_SCHEMA_VERSION = "page-provenance-audit-v1"
_MAX_ANCHOR_CHARS = 256
_MIN_STABLE_CONTENT_CHARS = 16
_MIN_UNIQUE_CONTENT_CHARS = 8
_MAX_BOUNDARY_TOKEN_OFFSET = 4
_MAX_BOUNDARY_ANCHOR_TOKENS = 20
_LEXICAL_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class AuditIssue(StrEnum):
    """Machine-readable provenance failure categories."""

    PAPER_RECORD_MISSING = "paper_record_missing"
    PDF_NOT_FOUND = "pdf_not_found"
    PDF_LOAD_FAILED = "pdf_load_failed"
    PDF_CONTENT_HASH_MISMATCH = "pdf_content_hash_mismatch"
    PAGE_COUNT_MISMATCH = "page_count_mismatch"
    PAGE_RANGE_INVALID = "page_range_invalid"
    START_ANCHOR_MISSING = "start_anchor_missing"
    END_ANCHOR_MISSING = "end_anchor_missing"


class AnchorCheck(BaseModel):
    """Evidence that one chunk edge occurs on one physical PDF page."""

    edge: Literal["start", "end"]
    page_number: int = Field(ge=1)
    matched: bool
    anchor: str = ""
    matched_content_chars: int = Field(ge=0)
    required_content_chars: int = Field(ge=0)
    matching_page_count: int = Field(default=0, ge=0)
    stability_rule: Literal["long", "unique_short", "boundary_position", "none"] = "none"
    token_offset_from_boundary: int | None = Field(default=None, ge=0)


class ChunkProvenanceResult(BaseModel):
    """Audit result for one canonical chunk."""

    chunk_id: str
    paper_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    pdf_page_count: int | None = Field(default=None, ge=1)
    physical_range_valid: bool
    start_anchor: AnchorCheck | None = None
    end_anchor: AnchorCheck | None = None
    issues: list[AuditIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues


class ProvenanceAuditSummary(BaseModel):
    """Stable aggregate suitable for CI and acceptance evidence."""

    corpus_fingerprint: str
    pdf_fingerprint: str
    papers_referenced: int = Field(ge=0)
    papers_loaded: int = Field(ge=0)
    chunks_checked: int = Field(ge=0)
    chunks_passed: int = Field(ge=0)
    chunks_failed: int = Field(ge=0)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    passed: bool


class ProvenanceAuditReport(BaseModel):
    """Complete deterministic audit artifact."""

    schema_version: Literal["page-provenance-audit-v1"] = _SCHEMA_VERSION
    summary: ProvenanceAuditSummary
    results: list[ChunkProvenanceResult]


def _normalize_content(text: str) -> str:
    """Normalize only representation details, never semantic content."""

    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\u00ad", "")
    value = value.translate(str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-"}))
    return " ".join(value.casefold().split())


def _content_char_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _trim_anchor(text: str) -> str:
    """Remove punctuation created by cutting a bounded display anchor."""

    start = 0
    end = len(text)
    while start < end and not text[start].isalnum():
        start += 1
    while end > start and not text[end - 1].isalnum():
        end -= 1
    return text[start:end]


def _lexical_tokens(text: str) -> list[str]:
    return _LEXICAL_TOKEN_RE.findall(text)


def _sequence_positions(haystack: list[str], needle: list[str]) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    return [
        index
        for index in range(len(haystack) - len(needle) + 1)
        if haystack[index : index + len(needle)] == needle
    ]


def _matching_physical_pages(
    anchor: str,
    normalized_page_variants: list[list[str]],
) -> int:
    return sum(
        any(anchor in variant for variant in variants) for variants in normalized_page_variants
    )


def _boundary_position_anchor(
    chunk: str,
    *,
    boundary_page_variants: list[str],
    normalized_page_variants: list[list[str]],
    edge: Literal["start", "end"],
) -> tuple[str, int, int, int] | None:
    """Return a complete token anchored at the physical page edge.

    Token position makes a one-word/table-label boundary fragment objective
    even when its text is common elsewhere.  This is needed for legitimate
    windows that contain only the first word of a new page or the final page
    number of the previous page.  The four-token tolerance accounts for a
    heading split into separate PDF text lines; it is deliberately too narrow
    to turn a range-interior occurrence into a boundary anchor.
    """

    chunk_tokens = _lexical_tokens(chunk)
    for length in range(min(_MAX_BOUNDARY_ANCHOR_TOKENS, len(chunk_tokens)), 0, -1):
        candidate_tokens = chunk_tokens[:length] if edge == "start" else chunk_tokens[-length:]
        if not any(any(character.isalnum() for character in token) for token in candidate_tokens):
            continue
        for variant in boundary_page_variants:
            page_tokens = _lexical_tokens(variant)
            for position in _sequence_positions(page_tokens, candidate_tokens):
                offset = len(page_tokens) - (position + length) if edge == "start" else position
                if offset > _MAX_BOUNDARY_TOKEN_OFFSET:
                    continue
                anchor = " ".join(candidate_tokens)
                page_matches = sum(
                    any(
                        _sequence_positions(_lexical_tokens(page_variant), candidate_tokens)
                        for page_variant in variants
                    )
                    for variants in normalized_page_variants
                )
                return anchor, _content_char_count(anchor), page_matches, offset

    # A production tokenizer may cut at a subword boundary (for example, a
    # window can contain just "K" from page-leading "Keheng").  Exact
    # character alignment at offset zero still proves physical ownership; an
    # arbitrary substring elsewhere on the page does not.
    limit = min(32, len(chunk))
    for length in range(limit, 0, -1):
        candidate = chunk[:length] if edge == "start" else chunk[-length:]
        if _content_char_count(candidate) < 1:
            continue
        aligned = any(
            variant.endswith(candidate) if edge == "start" else variant.startswith(candidate)
            for variant in boundary_page_variants
        )
        if aligned:
            return (
                candidate,
                _content_char_count(candidate),
                _matching_physical_pages(candidate, normalized_page_variants),
                0,
            )
    return None


def _ingestion_page_text_variants(pages: list[PaperPage]) -> dict[int, list[str]]:
    """Return raw-cleaned and heading-stripped page views used by ingestion."""

    # Avoid duplicating section parsing logic: chunks are built from the
    # SectionPageText projection, where detected headings are metadata rather
    # than body text.  Keeping the raw cleaned view as a second variant covers
    # layout fragments that section parsing intentionally omits.
    page_parts: dict[int, list[str]] = defaultdict(list)
    for section in pages_to_sections(pages):
        for item in section.page_texts:
            page_parts[item.page_number].append(item.text)

    variants: dict[int, list[str]] = {}
    for page in pages:
        page_number = page.page_number
        candidates = [page.text]
        projected = "\n\n".join(page_parts[page_number]).strip()
        if projected and projected != candidates[0]:
            candidates.append(projected)
        variants[page_number] = candidates
    return variants


def _edge_anchor_check(
    chunk_text: str,
    *,
    boundary_page_variants: list[str],
    normalized_page_variants: list[list[str]],
    page_number: int,
    edge: Literal["start", "end"],
) -> AnchorCheck:
    """Find the longest bounded chunk-edge substring on a boundary page.

    Exact normalized substrings are preferable to fuzzy token overlap here:
    the chunk was produced from these same extracted pages, and fuzzy overlap
    could let a common phrase on the wrong page validate an over-wide range.
    A long exact anchor is intrinsically stable.  A shorter anchor is accepted
    when it is unique within that PDF.  Finally, a complete token (or exact
    tokenizer subword) at the physical page edge is position-anchored.  This
    handles genuine one-token cross-page windows without accepting a common
    word found somewhere in the range interior.
    """

    chunk = _normalize_content(chunk_text)
    available_content = _content_char_count(chunk)
    required = min(_MIN_UNIQUE_CONTENT_CHARS, available_content)
    limit = min(_MAX_ANCHOR_CHARS, len(chunk))
    matched_anchor = ""
    matched_content = 0
    matching_page_count = 0
    stability_rule: Literal["long", "unique_short", "boundary_position", "none"] = "none"
    token_offset: int | None = None

    for length in range(limit, 0, -1):
        candidate = chunk[:length] if edge == "start" else chunk[-length:]
        candidate = _trim_anchor(candidate)
        if not candidate:
            continue
        content_chars = _content_char_count(candidate)
        if content_chars < required:
            break
        if any(candidate in page for page in boundary_page_variants):
            page_matches = _matching_physical_pages(candidate, normalized_page_variants)
            is_long = content_chars >= _MIN_STABLE_CONTENT_CHARS
            is_unique_short = page_matches == 1
            if is_long or is_unique_short:
                matched_anchor = candidate
                matched_content = content_chars
                matching_page_count = page_matches
                stability_rule = "long" if is_long else "unique_short"
                break

    if stability_rule == "none":
        positioned = _boundary_position_anchor(
            chunk,
            boundary_page_variants=boundary_page_variants,
            normalized_page_variants=normalized_page_variants,
            edge=edge,
        )
        if positioned is not None:
            matched_anchor, matched_content, matching_page_count, token_offset = positioned
            stability_rule = "boundary_position"

    return AnchorCheck(
        edge=edge,
        page_number=page_number,
        matched=required > 0 and stability_rule != "none",
        anchor=matched_anchor,
        matched_content_chars=matched_content,
        required_content_chars=required,
        matching_page_count=matching_page_count,
        stability_rule=stability_rule,
        token_offset_from_boundary=token_offset,
    )


def _resolve_pdf_path(paper: Paper, papers_dir: Path) -> Path | None:
    """Resolve relocatable canonical paths without changing stored records."""

    stored = Path(paper.pdf_path)
    candidates = [papers_dir / stored.name, stored]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _failed_result(
    chunk: Chunk,
    issue: AuditIssue,
    *,
    pdf_page_count: int | None = None,
) -> ChunkProvenanceResult:
    return ChunkProvenanceResult(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        pdf_page_count=pdf_page_count,
        physical_range_valid=False,
        issues=[issue],
    )


def audit_page_provenance(
    *,
    processed_dir: Path | str = Path("data/processed"),
    papers_dir: Path | str = Path("data/papers"),
) -> ProvenanceAuditReport:
    """Audit every canonical chunk against the physical PDF boundary pages."""

    processed_root = Path(processed_dir)
    pdf_root = Path(papers_dir)
    chunks_path = processed_root / "chunks.jsonl"
    papers_path = processed_root / "papers.jsonl"
    if not chunks_path.is_file():
        raise FileNotFoundError(f"canonical chunk store missing: {chunks_path}")
    if not papers_path.is_file():
        raise FileNotFoundError(f"canonical paper store missing: {papers_path}")
    chunks = JsonlRepository(chunks_path, Chunk).read_all()
    papers = JsonlRepository(papers_path, Paper).read_all()
    if not chunks:
        raise ValueError(f"canonical chunk store is empty: {chunks_path}")
    if not papers:
        raise ValueError(f"canonical paper store is empty: {papers_path}")
    papers_by_id = {paper.paper_id: paper for paper in papers}
    chunks_by_paper: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_paper[chunk.paper_id].append(chunk)

    results: list[ChunkProvenanceResult] = []
    papers_loaded = 0
    loaded_pdf_hashes: dict[str, str] = {}

    for paper_id in sorted(chunks_by_paper):
        paper_chunks = chunks_by_paper[paper_id]
        paper = papers_by_id.get(paper_id)
        if paper is None:
            results.extend(
                _failed_result(chunk, AuditIssue.PAPER_RECORD_MISSING) for chunk in paper_chunks
            )
            continue

        pdf_path = _resolve_pdf_path(paper, pdf_root)
        if pdf_path is None:
            results.extend(
                _failed_result(chunk, AuditIssue.PDF_NOT_FOUND) for chunk in paper_chunks
            )
            continue

        try:
            actual_pdf_hash = file_content_hash(pdf_path)
            raw_pages, _ = load_pages(paper_id, pdf_path)
            pages = strip_headers_footers(raw_pages)
        except (PDFLoadError, OSError, ValueError):
            results.extend(
                _failed_result(chunk, AuditIssue.PDF_LOAD_FAILED) for chunk in paper_chunks
            )
            continue

        papers_loaded += 1
        loaded_pdf_hashes[paper_id] = actual_pdf_hash
        page_count = len(pages)
        page_text_variants = _ingestion_page_text_variants(pages)
        normalized_variants_by_page = {
            page_number: [_normalize_content(text) for text in variants]
            for page_number, variants in page_text_variants.items()
        }
        normalized_page_variants = [
            normalized_variants_by_page[page_number]
            for page_number in sorted(normalized_variants_by_page)
        ]
        count_mismatch = paper.page_count is not None and paper.page_count != page_count
        hash_mismatch = actual_pdf_hash != paper.content_hash

        for chunk in paper_chunks:
            issues: list[AuditIssue] = []
            if count_mismatch:
                issues.append(AuditIssue.PAGE_COUNT_MISMATCH)
            if hash_mismatch:
                issues.append(AuditIssue.PDF_CONTENT_HASH_MISMATCH)

            range_valid = (
                chunk.page_start >= 1
                and chunk.page_end >= chunk.page_start
                and chunk.page_end <= page_count
            )
            if not range_valid:
                issues.append(AuditIssue.PAGE_RANGE_INVALID)
                results.append(
                    ChunkProvenanceResult(
                        chunk_id=chunk.chunk_id,
                        paper_id=chunk.paper_id,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        pdf_page_count=page_count,
                        physical_range_valid=False,
                        issues=issues,
                    )
                )
                continue

            start_anchor = _edge_anchor_check(
                chunk.text,
                boundary_page_variants=normalized_variants_by_page[chunk.page_start],
                normalized_page_variants=normalized_page_variants,
                page_number=chunk.page_start,
                edge="start",
            )
            end_anchor = _edge_anchor_check(
                chunk.text,
                boundary_page_variants=normalized_variants_by_page[chunk.page_end],
                normalized_page_variants=normalized_page_variants,
                page_number=chunk.page_end,
                edge="end",
            )
            if not start_anchor.matched:
                issues.append(AuditIssue.START_ANCHOR_MISSING)
            if not end_anchor.matched:
                issues.append(AuditIssue.END_ANCHOR_MISSING)
            results.append(
                ChunkProvenanceResult(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    pdf_page_count=page_count,
                    physical_range_valid=True,
                    start_anchor=start_anchor,
                    end_anchor=end_anchor,
                    issues=issues,
                )
            )

    # Stable canonical order makes independently generated JSON byte-comparable.
    results.sort(key=lambda result: (result.paper_id, result.page_start, result.chunk_id))
    issue_counts = Counter(issue.value for result in results for issue in result.issues)
    chunks_passed = sum(result.passed for result in results)
    chunks_failed = len(results) - chunks_passed
    pdf_hash_material = "\n".join(
        f"{paper_id}:{digest}" for paper_id, digest in sorted(loaded_pdf_hashes.items())
    )
    summary = ProvenanceAuditSummary(
        corpus_fingerprint=corpus_fingerprint(chunks),
        pdf_fingerprint=content_hash(pdf_hash_material, length=32),
        papers_referenced=len(chunks_by_paper),
        papers_loaded=papers_loaded,
        chunks_checked=len(results),
        chunks_passed=chunks_passed,
        chunks_failed=chunks_failed,
        issue_counts=dict(sorted(issue_counts.items())),
        passed=chunks_failed == 0,
    )
    return ProvenanceAuditReport(summary=summary, results=results)


def format_summary(report: ProvenanceAuditReport, *, failure_limit: int = 20) -> str:
    """Render a concise human-readable summary from the structured report."""

    summary = report.summary
    lines = [
        "Canonical page-provenance audit",
        f"corpus_fingerprint: {summary.corpus_fingerprint}",
        f"pdf_fingerprint: {summary.pdf_fingerprint}",
        f"papers: {summary.papers_loaded}/{summary.papers_referenced} loaded",
        (
            f"chunks: {summary.chunks_passed}/{summary.chunks_checked} passed; "
            f"{summary.chunks_failed} failed"
        ),
        f"result: {'PASS' if summary.passed else 'FAIL'}",
    ]
    if summary.issue_counts:
        rendered = ", ".join(f"{key}={value}" for key, value in summary.issue_counts.items())
        lines.append(f"issues: {rendered}")
    failures = [result for result in report.results if not result.passed]
    for result in failures[:failure_limit]:
        issues = ",".join(issue.value for issue in result.issues)
        lines.append(
            f"- {result.chunk_id} ({result.paper_id} "
            f"p.{result.page_start}-{result.page_end}): {issues}"
        )
    if len(failures) > failure_limit:
        lines.append(f"- ... {len(failures) - failure_limit} more failures (use --json)")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--papers-dir", type=Path, default=Path("data/papers"))
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete Pydantic-validated JSON report instead of a summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = audit_page_provenance(
            processed_dir=args.processed_dir,
            papers_dir=args.papers_dir,
        )
    except (OSError, ValueError) as exc:
        print(f"page-provenance audit could not start: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(format_summary(report))
    return 0 if report.summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

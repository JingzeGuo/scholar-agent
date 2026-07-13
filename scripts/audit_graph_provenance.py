#!/usr/bin/env python3
"""Read-only audit of graph-relation provenance against physical PDF pages.

The graph build localizes every relation evidence span to one or more physical
PDF pages.  This command independently reloads the canonical paper, chunk, and
relation stores and proves that the persisted range is both truthful and
minimal.  In particular, a broad range such as pages 14--16 is rejected when
the complete span already occurs in the smaller contiguous window 15--16.

Exit codes are stable for automation: 0 means every relation passed, 1 means
the audit ran and found provenance failures, and 2 means the audit inputs could
not be parsed or validated.  The command never writes to the corpus.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

from scholar_agent.graph.evidence import normalize_span
from scholar_agent.ids import content_hash
from scholar_agent.ingestion.headers import strip_headers_footers
from scholar_agent.ingestion.loader import PDFLoadError, file_content_hash, load_pages
from scholar_agent.models.corpus import Chunk, Paper
from scholar_agent.models.graph import Relation
from scholar_agent.retrieval.chunk_store import corpus_fingerprint
from scholar_agent.storage.jsonl import JsonlRepository, JsonlRepositoryError

_SCHEMA_VERSION: Literal["graph-provenance-audit-v1"] = "graph-provenance-audit-v1"
T = TypeVar("T", bound=BaseModel)


class GraphProvenanceIssue(StrEnum):
    """Machine-readable relation provenance failure categories."""

    CHUNK_MISSING = "chunk_missing"
    PAPER_MISSING = "paper_missing"
    RELATION_CHUNK_PAPER_MISMATCH = "relation_chunk_paper_mismatch"
    PDF_NOT_FOUND = "pdf_not_found"
    PDF_LOAD_FAILED = "pdf_load_failed"
    PDF_CONTENT_HASH_MISMATCH = "pdf_content_hash_mismatch"
    PAPER_PAGE_COUNT_MISMATCH = "paper_page_count_mismatch"
    CHUNK_RANGE_OUTSIDE_PDF = "chunk_range_outside_pdf"
    PAGE_RANGE_OUTSIDE_CHUNK = "page_range_outside_chunk"
    PAGE_RANGE_OUTSIDE_PDF = "page_range_outside_pdf"
    EVIDENCE_SPAN_NOT_IN_CHUNK = "evidence_span_not_in_chunk"
    EVIDENCE_SPAN_NOT_FOUND = "evidence_span_not_found"
    STORED_WINDOW_MISSING_SPAN = "stored_window_missing_span"
    PAGE_RANGE_NOT_MINIMAL = "page_range_not_minimal"


class PhysicalPageRange(BaseModel):
    """Inclusive, one-indexed physical PDF page range."""

    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class RelationProvenanceResult(BaseModel):
    """Independent provenance verdict for one persisted graph relation."""

    relation_id: str
    paper_id: str
    chunk_id: str
    stored_page_start: int = Field(ge=1)
    stored_page_end: int = Field(ge=1)
    chunk_page_start: int | None = Field(default=None, ge=1)
    chunk_page_end: int | None = Field(default=None, ge=1)
    pdf_page_count: int | None = Field(default=None, ge=1)
    evidence_span_hash: str
    span_found_in_chunk: bool | None = None
    stored_window_contains_span: bool | None = None
    stored_range_is_minimal: bool = False
    minimal_candidate_ranges: list[PhysicalPageRange] = Field(default_factory=list)
    issues: list[GraphProvenanceIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues


class GraphProvenanceSummary(BaseModel):
    """Aggregate acceptance evidence suitable for CI and release reports."""

    corpus_fingerprint: str
    relations_fingerprint: str
    papers_referenced: int = Field(ge=0)
    papers_loaded: int = Field(ge=0)
    relations_checked: int = Field(ge=0)
    relations_passed: int = Field(ge=0)
    relations_failed: int = Field(ge=0)
    single_page_ranges: int = Field(ge=0)
    cross_page_ranges: int = Field(ge=0)
    minimal_ranges: int = Field(ge=0)
    missing_spans: int = Field(ge=0)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    passed: bool


class GraphProvenanceReport(BaseModel):
    """Complete deterministic, Pydantic-validated audit report."""

    schema_version: Literal["graph-provenance-audit-v1"] = _SCHEMA_VERSION
    summary: GraphProvenanceSummary
    results: list[RelationProvenanceResult]


class _LoadedPaper(BaseModel):
    """Internal normalized physical-page view, loaded once per PDF."""

    page_count: int = Field(ge=1)
    normalized_pages: dict[int, str]
    issues: list[GraphProvenanceIssue] = Field(default_factory=list)


def _unique_by_id(
    rows: list[T],
    *,
    field_name: str,
    source: Path,
) -> dict[str, T]:
    """Index a validated store while treating duplicate identities as damage."""

    indexed: dict[str, T] = {}
    for row in rows:
        value = getattr(row, field_name, None)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{source}: invalid {field_name}")
        if value in indexed:
            raise ValueError(f"{source}: duplicate {field_name}={value!r}")
        indexed[value] = row
    return indexed


def _resolve_pdf_path(paper: Paper, papers_dir: Path) -> Path | None:
    """Resolve a relocatable corpus path without changing the paper record."""

    stored = Path(paper.pdf_path)
    for candidate in (papers_dir / stored.name, stored):
        if candidate.is_file():
            return candidate
    return None


def _load_physical_paper(paper: Paper, papers_dir: Path) -> _LoadedPaper | None:
    """Load and normalize a real PDF using the ingestion cleanup semantics."""

    pdf_path = _resolve_pdf_path(paper, papers_dir)
    if pdf_path is None:
        return None
    raw_pages, _ = load_pages(paper.paper_id, pdf_path)
    pages = strip_headers_footers(raw_pages)
    issues: list[GraphProvenanceIssue] = []
    if file_content_hash(pdf_path) != paper.content_hash:
        issues.append(GraphProvenanceIssue.PDF_CONTENT_HASH_MISMATCH)
    if paper.page_count is not None and paper.page_count != len(pages):
        issues.append(GraphProvenanceIssue.PAPER_PAGE_COUNT_MISMATCH)
    return _LoadedPaper(
        page_count=len(pages),
        normalized_pages={page.page_number: normalize_span(page.text) for page in pages},
        issues=issues,
    )


def _window_text(normalized_pages: dict[int, str], start: int, end: int) -> str | None:
    """Return one normalized continuous physical page window, if it exists."""

    values: list[str] = []
    for page_number in range(start, end + 1):
        text = normalized_pages.get(page_number)
        if text is None:
            return None
        values.append(text)
    return " ".join(values)


def _minimal_windows(
    normalized_span: str,
    normalized_pages: dict[int, str],
    *,
    page_start: int,
    page_end: int,
) -> list[PhysicalPageRange]:
    """Find every shortest continuous physical page window containing span."""

    if not normalized_span or page_start < 1 or page_end < page_start:
        return []
    width_limit = page_end - page_start + 1
    for width in range(1, width_limit + 1):
        matches: list[PhysicalPageRange] = []
        for start in range(page_start, page_end - width + 2):
            end = start + width - 1
            window = _window_text(normalized_pages, start, end)
            if window is not None and normalized_span in window:
                matches.append(PhysicalPageRange(page_start=start, page_end=end))
        if matches:
            return matches
    return []


def _relations_fingerprint(relations: list[Relation]) -> str:
    material = "\n".join(
        relation.model_dump_json()
        for relation in sorted(relations, key=lambda item: item.relation_id)
    )
    return content_hash(material, length=32)


def _base_result(relation: Relation, chunk: Chunk | None) -> RelationProvenanceResult:
    return RelationProvenanceResult(
        relation_id=relation.relation_id,
        paper_id=relation.paper_id,
        chunk_id=relation.chunk_id,
        stored_page_start=relation.page_number,
        stored_page_end=relation.page_end or relation.page_number,
        chunk_page_start=chunk.page_start if chunk is not None else None,
        chunk_page_end=chunk.page_end if chunk is not None else None,
        evidence_span_hash=content_hash(normalize_span(relation.evidence_span), length=16),
    )


def _audit_relation(
    relation: Relation,
    *,
    chunks_by_id: dict[str, Chunk],
    papers_by_id: dict[str, Paper],
    loaded_papers: dict[str, _LoadedPaper | GraphProvenanceIssue],
) -> RelationProvenanceResult:
    chunk = chunks_by_id.get(relation.chunk_id)
    result = _base_result(relation, chunk)
    issues: list[GraphProvenanceIssue] = []
    if chunk is None:
        issues.append(GraphProvenanceIssue.CHUNK_MISSING)
    paper = papers_by_id.get(relation.paper_id)
    if paper is None:
        issues.append(GraphProvenanceIssue.PAPER_MISSING)
    if chunk is not None and chunk.paper_id != relation.paper_id:
        issues.append(GraphProvenanceIssue.RELATION_CHUNK_PAPER_MISMATCH)
    if chunk is None or paper is None or chunk.paper_id != relation.paper_id:
        return result.model_copy(update={"issues": issues})

    normalized_span = normalize_span(relation.evidence_span)
    span_found_in_chunk = normalized_span in normalize_span(chunk.text)
    if not span_found_in_chunk:
        issues.append(GraphProvenanceIssue.EVIDENCE_SPAN_NOT_IN_CHUNK)

    loaded = loaded_papers.get(relation.paper_id)
    if loaded is None:
        issues.append(GraphProvenanceIssue.PDF_NOT_FOUND)
        return result.model_copy(
            update={"span_found_in_chunk": span_found_in_chunk, "issues": issues}
        )
    if isinstance(loaded, GraphProvenanceIssue):
        issues.append(loaded)
        return result.model_copy(
            update={"span_found_in_chunk": span_found_in_chunk, "issues": issues}
        )

    issues.extend(loaded.issues)
    stored_start = relation.page_number
    stored_end = relation.page_end or stored_start
    range_in_chunk = chunk.page_start <= stored_start <= stored_end <= chunk.page_end
    if not range_in_chunk:
        issues.append(GraphProvenanceIssue.PAGE_RANGE_OUTSIDE_CHUNK)
    range_in_pdf = 1 <= stored_start <= stored_end <= loaded.page_count
    if not range_in_pdf:
        issues.append(GraphProvenanceIssue.PAGE_RANGE_OUTSIDE_PDF)
    chunk_range_in_pdf = 1 <= chunk.page_start <= chunk.page_end <= loaded.page_count
    if not chunk_range_in_pdf:
        issues.append(GraphProvenanceIssue.CHUNK_RANGE_OUTSIDE_PDF)

    candidates: list[PhysicalPageRange] = []
    if chunk_range_in_pdf:
        candidates = _minimal_windows(
            normalized_span,
            loaded.normalized_pages,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )
        if not candidates:
            issues.append(GraphProvenanceIssue.EVIDENCE_SPAN_NOT_FOUND)

    stored_contains: bool | None = None
    if range_in_pdf:
        stored_text = _window_text(loaded.normalized_pages, stored_start, stored_end)
        stored_contains = stored_text is not None and normalized_span in stored_text
        if not stored_contains:
            issues.append(GraphProvenanceIssue.STORED_WINDOW_MISSING_SPAN)

    stored_range = PhysicalPageRange(page_start=stored_start, page_end=stored_end)
    stored_is_minimal = stored_range in candidates
    if candidates and not stored_is_minimal:
        issues.append(GraphProvenanceIssue.PAGE_RANGE_NOT_MINIMAL)

    return result.model_copy(
        update={
            "pdf_page_count": loaded.page_count,
            "span_found_in_chunk": span_found_in_chunk,
            "stored_window_contains_span": stored_contains,
            "stored_range_is_minimal": stored_is_minimal,
            "minimal_candidate_ranges": candidates,
            "issues": list(dict.fromkeys(issues)),
        }
    )


def audit_graph_provenance(
    *,
    processed_dir: Path | str = Path("data/processed"),
    papers_dir: Path | str = Path("data/papers"),
) -> GraphProvenanceReport:
    """Audit every canonical relation against chunks, papers, and real PDFs."""

    processed_root = Path(processed_dir)
    pdf_root = Path(papers_dir)
    papers_path = processed_root / "papers.jsonl"
    chunks_path = processed_root / "chunks.jsonl"
    relations_path = processed_root / "relations.jsonl"
    for path in (papers_path, chunks_path, relations_path):
        if not path.is_file():
            raise FileNotFoundError(f"required canonical store missing: {path}")

    papers = JsonlRepository(papers_path, Paper).read_all()
    chunks = JsonlRepository(chunks_path, Chunk).read_all()
    relations = JsonlRepository(relations_path, Relation).read_all()
    if not papers:
        raise ValueError(f"canonical paper store is empty: {papers_path}")
    if not chunks:
        raise ValueError(f"canonical chunk store is empty: {chunks_path}")
    if not relations:
        raise ValueError(f"canonical relation store is empty: {relations_path}")

    papers_by_id = _unique_by_id(papers, field_name="paper_id", source=papers_path)
    chunks_by_id = _unique_by_id(chunks, field_name="chunk_id", source=chunks_path)
    _unique_by_id(relations, field_name="relation_id", source=relations_path)

    referenced_paper_ids = sorted({relation.paper_id for relation in relations})
    loaded_papers: dict[str, _LoadedPaper | GraphProvenanceIssue] = {}
    for paper_id in referenced_paper_ids:
        paper = papers_by_id.get(paper_id)
        if paper is None:
            continue
        try:
            loaded = _load_physical_paper(paper, pdf_root)
        except (PDFLoadError, OSError, RuntimeError, ValueError):
            loaded_papers[paper_id] = GraphProvenanceIssue.PDF_LOAD_FAILED
            continue
        if loaded is not None:
            loaded_papers[paper_id] = loaded

    results = [
        _audit_relation(
            relation,
            chunks_by_id=chunks_by_id,
            papers_by_id=papers_by_id,
            loaded_papers=loaded_papers,
        )
        for relation in relations
    ]
    results.sort(key=lambda item: item.relation_id)
    issue_counts = Counter(issue.value for result in results for issue in result.issues)
    passed = sum(result.passed for result in results)
    minimal = sum(result.stored_range_is_minimal for result in results)
    single = sum(
        result.stored_range_is_minimal and result.stored_page_start == result.stored_page_end
        for result in results
    )
    cross = sum(
        result.stored_range_is_minimal and result.stored_page_start != result.stored_page_end
        for result in results
    )
    missing = sum(
        GraphProvenanceIssue.EVIDENCE_SPAN_NOT_FOUND in result.issues for result in results
    )
    summary = GraphProvenanceSummary(
        corpus_fingerprint=corpus_fingerprint(chunks),
        relations_fingerprint=_relations_fingerprint(relations),
        papers_referenced=len(referenced_paper_ids),
        papers_loaded=sum(isinstance(item, _LoadedPaper) for item in loaded_papers.values()),
        relations_checked=len(results),
        relations_passed=passed,
        relations_failed=len(results) - passed,
        single_page_ranges=single,
        cross_page_ranges=cross,
        minimal_ranges=minimal,
        missing_spans=missing,
        issue_counts=dict(sorted(issue_counts.items())),
        passed=passed == len(results),
    )
    return GraphProvenanceReport(summary=summary, results=results)


def format_summary(report: GraphProvenanceReport, *, failure_limit: int = 20) -> str:
    """Render a concise acceptance summary and bounded failure details."""

    summary = report.summary
    lines = [
        "Graph relation-provenance audit",
        f"corpus_fingerprint: {summary.corpus_fingerprint}",
        f"relations_fingerprint: {summary.relations_fingerprint}",
        f"papers: {summary.papers_loaded}/{summary.papers_referenced} loaded",
        (
            f"relations: {summary.relations_passed}/{summary.relations_checked} passed; "
            f"{summary.relations_failed} failed"
        ),
        (
            "provenance: "
            f"single={summary.single_page_ranges} cross={summary.cross_page_ranges} "
            f"minimal={summary.minimal_ranges} missing={summary.missing_spans}"
        ),
        f"result: {'PASS' if summary.passed else 'FAIL'}",
    ]
    if summary.issue_counts:
        rendered = ", ".join(f"{key}={value}" for key, value in summary.issue_counts.items())
        lines.append(f"issues: {rendered}")
    failures = [result for result in report.results if not result.passed]
    for result in failures[:failure_limit]:
        issue_text = ",".join(issue.value for issue in result.issues)
        lines.append(
            f"- {result.relation_id} ({result.paper_id}/{result.chunk_id} "
            f"p.{result.stored_page_start}-{result.stored_page_end}): {issue_text}"
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
        report = audit_graph_provenance(
            processed_dir=args.processed_dir,
            papers_dir=args.papers_dir,
        )
    except (JsonlRepositoryError, OSError, ValueError) as exc:
        print(f"graph-provenance audit could not start: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(format_summary(report))
    return 0 if report.summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

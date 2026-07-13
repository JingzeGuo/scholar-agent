"""Independent physical-PDF acceptance tests for graph provenance."""

from __future__ import annotations

import json
from pathlib import Path

import fitz
from scripts.audit_graph_provenance import (
    GraphProvenanceIssue,
    audit_graph_provenance,
    main,
)

from scholar_agent.ids import content_hash
from scholar_agent.ingestion.loader import file_content_hash
from scholar_agent.models.corpus import Chunk, Paper
from scholar_agent.models.graph import Relation, RelationType
from scholar_agent.storage.jsonl import JsonlRepository


def _write_three_page_pdf(path: Path) -> None:
    bodies = [
        "Single-page evidence is present here. Unrelated first-page context.",
        "Cross-page evidence starts on physical page two",
        "and finishes on physical page three. Final context follows.",
    ]
    document = fitz.open()
    for body in bodies:
        page = document.new_page(width=612, height=792)
        page.insert_text((72, 90), body, fontsize=11)
    document.save(path)
    document.close()


def _relation(
    relation_id: str,
    chunk: Chunk,
    evidence_span: str,
    *,
    page_start: int,
    page_end: int,
) -> Relation:
    return Relation(
        relation_id=relation_id,
        subject_surface="audit subject",
        object_surface="audit object",
        relation_type=RelationType.USES,
        evidence_span=evidence_span,
        paper_id=chunk.paper_id,
        chunk_id=chunk.chunk_id,
        page_number=page_start,
        page_end=page_end,
        confidence=0.9,
    )


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, list[Relation]]:
    processed = tmp_path / "processed"
    papers_dir = tmp_path / "papers"
    processed.mkdir()
    papers_dir.mkdir()
    pdf_path = papers_dir / "audit.pdf"
    _write_three_page_pdf(pdf_path)

    paper = Paper(
        paper_id="paper_audit",
        title="Graph provenance audit fixture",
        pdf_path=str(pdf_path),
        content_hash=file_content_hash(pdf_path),
        page_count=3,
    )
    single_text = "Single-page evidence is present here."
    single = Chunk(
        chunk_id="chunk_single",
        paper_id=paper.paper_id,
        text=single_text,
        page_start=1,
        page_end=1,
        token_count=6,
        content_hash=content_hash(single_text),
    )
    cross_span = (
        "Cross-page evidence starts on physical page two and finishes on physical page three."
    )
    missing_span = "Fabricated evidence absent from every physical page."
    cross_text = f"Unrelated first-page context. {cross_span} {missing_span}"
    cross = Chunk(
        chunk_id="chunk_cross",
        paper_id=paper.paper_id,
        text=cross_text,
        page_start=1,
        page_end=3,
        token_count=20,
        content_hash=content_hash(cross_text),
    )
    relations = [
        _relation("rel_single", single, single_text, page_start=1, page_end=1),
        _relation("rel_cross", cross, cross_span, page_start=2, page_end=3),
        _relation("rel_overwide", cross, cross_span, page_start=1, page_end=3),
        _relation("rel_missing", cross, missing_span, page_start=2, page_end=3),
    ]
    JsonlRepository(processed / "papers.jsonl", Paper).write_all([paper])
    JsonlRepository(processed / "chunks.jsonl", Chunk).write_all([single, cross])
    JsonlRepository(processed / "relations.jsonl", Relation).write_all(relations)
    return processed, papers_dir, relations


def test_audit_graph_provenance_detects_minimal_and_missing_ranges(tmp_path: Path) -> None:
    processed, papers_dir, _ = _build_fixture(tmp_path)

    report = audit_graph_provenance(processed_dir=processed, papers_dir=papers_dir)

    assert report.summary.relations_checked == 4
    assert report.summary.relations_passed == 2
    assert report.summary.relations_failed == 2
    assert report.summary.single_page_ranges == 1
    assert report.summary.cross_page_ranges == 1
    assert report.summary.minimal_ranges == 2
    assert report.summary.missing_spans == 1
    assert not report.summary.passed

    by_id = {result.relation_id: result for result in report.results}
    assert by_id["rel_single"].passed
    assert by_id["rel_single"].stored_range_is_minimal
    assert by_id["rel_cross"].passed
    assert [
        candidate.model_dump() for candidate in by_id["rel_cross"].minimal_candidate_ranges
    ] == [{"page_start": 2, "page_end": 3}]

    overwide = by_id["rel_overwide"]
    assert overwide.stored_window_contains_span
    assert not overwide.stored_range_is_minimal
    assert overwide.issues == [GraphProvenanceIssue.PAGE_RANGE_NOT_MINIMAL]

    missing = by_id["rel_missing"]
    assert missing.span_found_in_chunk
    assert not missing.stored_window_contains_span
    assert GraphProvenanceIssue.EVIDENCE_SPAN_NOT_FOUND in missing.issues
    assert GraphProvenanceIssue.STORED_WINDOW_MISSING_SPAN in missing.issues


def test_audit_graph_provenance_cli_exit_codes_and_json(
    tmp_path: Path,
    capsys: object,
) -> None:
    processed, papers_dir, relations = _build_fixture(tmp_path)
    args = ["--processed-dir", str(processed), "--papers-dir", str(papers_dir)]

    assert main(args) == 1
    JsonlRepository(processed / "relations.jsonl", Relation).write_all(relations[:2])
    assert main([*args, "--json"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    json_start = captured.out.index("{")
    payload = json.loads(captured.out[json_start:])
    assert payload["summary"]["passed"] is True
    assert payload["summary"]["relations_passed"] == 2

    (processed / "relations.jsonl").write_text("{broken-json\n", encoding="utf-8")
    assert main(args) == 2

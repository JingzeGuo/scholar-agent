"""Tests for the read-only canonical chunk page-provenance audit."""

from __future__ import annotations

import json
from pathlib import Path

import fitz
from scripts.audit_page_provenance import (
    AuditIssue,
    ProvenanceAuditReport,
    audit_page_provenance,
    format_summary,
    main,
)

from scholar_agent.ids import content_hash, make_chunk_id
from scholar_agent.ingestion import tokens as ingestion_tokens
from scholar_agent.ingestion.chunker import chunk_sections
from scholar_agent.ingestion.headers import strip_headers_footers
from scholar_agent.ingestion.loader import load_pages
from scholar_agent.models.corpus import Chunk, Paper
from scholar_agent.models.ingestion import SectionBlock, SectionPageText
from scholar_agent.storage.jsonl import JsonlRepository


def _write_multipage_pdf(path: Path) -> None:
    doc = fitz.open()
    bodies = [
        [
            "STARTANCHOR alpha retrieval evidence begins on physical page one.",
            "The first boundary contains distinctive canonical source content.",
        ],
        [
            "Middle continuation carries graph retrieval observations.",
            "ENDANCHOR omega verification evidence ends on physical page two.",
        ],
        [
            "Unrelated appendix material belongs only to physical page three.",
            "This page must not validate an over-wide chunk range.",
        ],
    ]
    for page_number, lines in enumerate(bodies, start=1):
        page = doc.new_page()
        page.insert_text((72, 48), f"Repeated provenance header {page_number}", fontsize=10)
        for offset, line in enumerate(lines):
            page.insert_text((72, 100 + offset * 24), line, fontsize=11)
        page.insert_text((72, 800), f"Repeated provenance footer {page_number}", fontsize=9)
    doc.save(path)
    doc.close()


def _chunk(paper_id: str, text: str, page_start: int, page_end: int) -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(
            paper_id,
            page_start=page_start,
            page_end=page_end,
            text=text,
        ),
        paper_id=paper_id,
        text=text,
        page_start=page_start,
        page_end=page_end,
        token_count=len(text.split()),
        content_hash=content_hash(text),
    )


def _write_canonical_stores(
    processed_dir: Path,
    *,
    pdf: Path,
    chunks: list[Chunk],
) -> None:
    processed_dir.mkdir()
    paper = Paper(
        paper_id="paper_provenance",
        title="Synthetic provenance paper",
        pdf_path=str(pdf),
        content_hash=content_hash(pdf.read_bytes()),
        page_count=3,
    )
    JsonlRepository(processed_dir / "papers.jsonl", Paper).write_all([paper])
    JsonlRepository(processed_dir / "chunks.jsonl", Chunk).write_all(chunks)


def test_audit_accepts_true_boundary_anchors_and_short_chunks(tmp_path: Path, capsys) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf = papers_dir / "paper.pdf"
    _write_multipage_pdf(pdf)
    pages, _ = load_pages("paper_provenance", pdf)
    cleaned = strip_headers_footers(pages)
    spanning_text = f"{cleaned[0].text}\n\n{cleaned[1].text}"
    short_text = "ENDANCHOR"
    one_token_boundary_text = f"{cleaned[0].text}\n\nMiddle"
    chunks = [
        _chunk("paper_provenance", spanning_text, 1, 2),
        _chunk("paper_provenance", short_text, 2, 2),
        _chunk("paper_provenance", one_token_boundary_text, 1, 2),
    ]
    processed_dir = tmp_path / "processed"
    _write_canonical_stores(processed_dir, pdf=pdf, chunks=chunks)

    report = audit_page_provenance(processed_dir=processed_dir, papers_dir=papers_dir)

    assert isinstance(report, ProvenanceAuditReport)
    assert report.summary.passed
    assert report.summary.chunks_checked == 3
    assert report.summary.chunks_passed == 3
    spanning = next(
        result
        for result in report.results
        if result.page_start != result.page_end
        and result.end_anchor is not None
        and result.end_anchor.stability_rule == "long"
    )
    assert spanning.start_anchor is not None and spanning.start_anchor.matched
    assert spanning.end_anchor is not None and spanning.end_anchor.matched
    assert "startanchor" in spanning.start_anchor.anchor
    assert "endanchor" in spanning.end_anchor.anchor
    boundary_short = next(
        result
        for result in report.results
        if result.end_anchor is not None and result.end_anchor.stability_rule == "boundary_position"
    )
    assert boundary_short.end_anchor is not None
    assert boundary_short.end_anchor.anchor.casefold() == "middle"
    assert boundary_short.end_anchor.token_offset_from_boundary == 0
    assert "chunks: 3/3 passed" in format_summary(report)

    exit_code = main(
        [
            "--processed-dir",
            str(processed_dir),
            "--papers-dir",
            str(papers_dir),
            "--json",
        ]
    )
    assert exit_code == 0
    emitted = ProvenanceAuditReport.model_validate(json.loads(capsys.readouterr().out))
    assert emitted.summary.passed


def test_audit_rejects_overwide_and_physically_invalid_ranges(
    tmp_path: Path,
    capsys,
) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf = papers_dir / "paper.pdf"
    _write_multipage_pdf(pdf)
    pages, _ = load_pages("paper_provenance", pdf)
    cleaned = strip_headers_footers(pages)
    text_from_pages_one_and_two = f"{cleaned[0].text}\n\n{cleaned[1].text}"
    chunks = [
        # Page three is physically real but contains no end-of-chunk content.
        _chunk("paper_provenance", text_from_pages_one_and_two, 1, 3),
        # Page four does not exist at all.
        _chunk("paper_provenance", "nonexistent boundary", 4, 4),
    ]
    processed_dir = tmp_path / "processed"
    _write_canonical_stores(processed_dir, pdf=pdf, chunks=chunks)

    report = audit_page_provenance(processed_dir=processed_dir, papers_dir=papers_dir)

    assert not report.summary.passed
    assert report.summary.chunks_failed == 2
    overwide = next(result for result in report.results if result.page_end == 3)
    assert overwide.physical_range_valid
    assert overwide.start_anchor is not None and overwide.start_anchor.matched
    assert overwide.end_anchor is not None and not overwide.end_anchor.matched
    assert overwide.issues == [AuditIssue.END_ANCHOR_MISSING]
    outside_pdf = next(result for result in report.results if result.page_end == 4)
    assert not outside_pdf.physical_range_valid
    assert outside_pdf.issues == [AuditIssue.PAGE_RANGE_INVALID]

    assert main(["--processed-dir", str(processed_dir), "--papers-dir", str(papers_dir)]) == 1
    output = capsys.readouterr().out
    assert "result: FAIL" in output
    assert "end_anchor_missing" in output


def test_chunker_separator_token_does_not_claim_the_following_page(monkeypatch) -> None:
    encoding = ingestion_tokens._LocalReversibleEncoding()
    monkeypatch.setattr(ingestion_tokens, "get_encoding", lambda _name="cl100k_base": encoding)
    page_one = " ".join(f"alpha{index}" for index in range(49))
    page_two = " ".join(f"beta{index}" for index in range(20))
    section = SectionBlock(
        title="Method",
        page_start=1,
        page_end=2,
        text=f"{page_one}\n\n{page_two}",
        page_texts=[
            SectionPageText(page_number=1, text=page_one),
            SectionPageText(page_number=2, text=page_two),
        ],
    )

    chunks = chunk_sections(
        "paper_separator",
        [section],
        target_tokens=50,
        overlap_tokens=0,
        min_tokens=1,
    )

    assert chunks[0].text == page_one
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)
    assert (chunks[1].page_start, chunks[1].page_end) == (2, 2)

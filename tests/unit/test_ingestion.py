"""Phase 2 ingestion tests: pages, chunking, empty/scanned flags, idempotency."""

from __future__ import annotations

from pathlib import Path

import fitz

from scholar_agent.config import ChunkingConfig
from scholar_agent.ids import content_hash
from scholar_agent.ingestion.chunker import chunk_sections
from scholar_agent.ingestion.headers import strip_headers_footers
from scholar_agent.ingestion.loader import load_pages
from scholar_agent.ingestion.pipeline import IngestionPipeline
from scholar_agent.ingestion.quality import assess_pages
from scholar_agent.ingestion.sections import is_section_heading, pages_to_sections
from scholar_agent.ingestion.tokens import count_tokens
from scholar_agent.models.corpus import (
    Chunk,
    CorpusManifestEntry,
    IngestionStatus,
    Paper,
    PaperPage,
)
from scholar_agent.models.ingestion import SectionBlock
from scholar_agent.storage.jsonl import JsonlRepository
from scholar_agent.storage.manifest import save_corpus_manifest


def _write_text_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        # Insert body text with a repeated header line
        page.insert_text((72, 50), "ScholarAgent Header v1", fontsize=10)
        y = 90
        for line in text.split("\n"):
            page.insert_text((72, y), line[:90], fontsize=11)
            y += 14
            if y > 750:
                break
        page.insert_text((72, 800), "Page footer ScholarAgent", fontsize=9)
    doc.save(path)
    doc.close()


def _write_empty_pdf(path: Path, *, n_pages: int = 2) -> None:
    doc = fitz.open()
    for _ in range(n_pages):
        doc.new_page()
    doc.save(path)
    doc.close()


def _write_scanned_like_pdf(path: Path) -> None:
    """Page with an image and almost no text → scanned_suspect."""
    doc = fitz.open()
    page = doc.new_page()
    # Tiny pixmap as image
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 64), 1)
    pix.clear_with(255)
    page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    doc.close()


def test_token_count_is_not_char_count() -> None:
    text = "retrieval augmented generation " * 20
    tokens = count_tokens(text)
    assert tokens > 0
    assert tokens < len(text)  # tokenizer merges subwords


def test_section_heading_detection() -> None:
    assert is_section_heading("1. Introduction")
    assert is_section_heading("Abstract")
    assert is_section_heading("RELATED WORK")
    assert not is_section_heading(
        "We propose a new retrieval method that improves accuracy significantly."
    )


def test_load_pages_preserves_page_numbers(tmp_path: Path) -> None:
    pdf = tmp_path / "multi.pdf"
    _write_text_pdf(
        pdf,
        [
            "Abstract\nThis is page one about RAG systems.",
            "1. Introduction\nThis is page two with more content about retrieval.",
            "2. Method\nThis is page three describing the approach in detail.",
        ],
    )
    pages, _ = load_pages("paper_test", pdf)
    assert len(pages) == 3
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "page one" in pages[0].text.lower()
    assert "page two" in pages[1].text.lower()
    assert "page three" in pages[2].text.lower()
    assert all(not p.is_empty for p in pages)


def test_header_footer_stripping(tmp_path: Path) -> None:
    pdf = tmp_path / "hdr.pdf"
    _write_text_pdf(
        pdf,
        [
            "Abstract\nBody content alpha unique.",
            "Introduction\nBody content beta unique.",
            "Method\nBody content gamma unique.",
        ],
    )
    pages, _ = load_pages("paper_hdr", pdf)
    cleaned = strip_headers_footers(pages)
    # Header/footer should be reduced or removed across pages
    joined = "\n".join(p.text for p in cleaned)
    assert "Body content alpha" in joined
    # Exact header string may still appear if detection threshold not met on short docs;
    # with 3 pages repeated edges should fire.
    header_hits = sum("ScholarAgent Header v1" in p.text for p in cleaned)
    assert header_hits <= 1


def test_chunker_token_aware_and_stable_ids() -> None:
    long_text = " ".join(["retrieval"] * 400)
    sections = [
        SectionBlock(title="1. Introduction", page_start=1, page_end=2, text=long_text),
        SectionBlock(title="2. Method", page_start=3, page_end=3, text="Short method text."),
    ]
    chunks_a = chunk_sections(
        "paper_x",
        sections,
        target_tokens=100,
        overlap_tokens=20,
        min_tokens=20,
    )
    chunks_b = chunk_sections(
        "paper_x",
        sections,
        target_tokens=100,
        overlap_tokens=20,
        min_tokens=20,
    )
    assert len(chunks_a) >= 2
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert all(c.paper_id == "paper_x" for c in chunks_a)
    assert all(c.page_start >= 1 for c in chunks_a)
    assert chunks_a[-1].section == "2. Method"


def test_empty_pdf_flagged(tmp_path: Path) -> None:
    pdf = tmp_path / "empty.pdf"
    _write_empty_pdf(pdf)
    pages, _ = load_pages("paper_empty", pdf)
    report = assess_pages("paper_empty", str(pdf), pages)
    assert report.is_empty_paper
    assert report.has_errors
    assert any(i.code == "empty_paper" for i in report.issues)


def test_scanned_pdf_flagged(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    _write_scanned_like_pdf(pdf)
    pages, _ = load_pages("paper_scan", pdf)
    assert pages[0].is_scanned_suspect or pages[0].is_empty
    report = assess_pages("paper_scan", str(pdf), pages)
    assert report.is_empty_paper or report.is_scanned_suspect


def test_pipeline_idempotent(tmp_path: Path, repo_root: Path) -> None:
    papers_dir = tmp_path / "papers"
    processed = tmp_path / "processed"
    papers_dir.mkdir()
    pdf = papers_dir / "demo.pdf"
    body = (
        "Abstract\n"
        "We study retrieval-augmented generation for knowledge intensive tasks.\n\n"
        "1. Introduction\n"
        + ("Dense retrieval and generation work together. " * 40)
        + "\n\n2. Method\n"
        + ("The model retrieves passages then conditions the decoder. " * 40)
    )
    _write_text_pdf(pdf, [body, body, "3. Conclusion\nWe conclude that RAG helps."])

    entry = CorpusManifestEntry(
        paper_id="paper_demo_1",
        title="Demo RAG Paper",
        authors=["Test Author"],
        year=2024,
        arxiv_id="9999.00001",
        pdf_filename="demo.pdf",
        source_url="https://example.com",
        topic_labels=["test"],
        content_hash=content_hash(pdf.read_bytes()),
        ingestion_status=IngestionStatus.PENDING,
    )
    manifest_path = tmp_path / "manifest.jsonl"
    save_corpus_manifest(manifest_path, [entry])

    pipeline = IngestionPipeline(
        papers_dir=papers_dir,
        processed_dir=processed,
        chunking=ChunkingConfig(target_tokens=120, overlap_tokens=20, min_tokens=20),
    )
    first = pipeline.ingest_entry(entry, force=False)
    assert first.status == IngestionStatus.INGESTED
    assert first.chunks
    assert first.paper is not None
    chunk_ids_1 = [c.chunk_id for c in first.chunks]

    second = pipeline.ingest_entry(first.entry, force=False)
    assert second.status == IngestionStatus.SKIPPED
    assert second.report.skipped is True

    # Force re-ingest yields same chunk IDs (stable content addressing)
    third = pipeline.ingest_entry(first.entry, force=True)
    assert third.status == IngestionStatus.INGESTED
    assert [c.chunk_id for c in third.chunks] == chunk_ids_1

    # Canonical stores
    papers = JsonlRepository(processed / "papers.jsonl", Paper).read_all()
    chunks = JsonlRepository(processed / "chunks.jsonl", Chunk).read_all()
    assert len(papers) == 1
    assert len(chunks) == len(chunk_ids_1)
    assert all(c.page_start <= c.page_end for c in chunks)


def test_empty_paper_not_indexed(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    processed = tmp_path / "processed"
    papers_dir.mkdir()
    pdf = papers_dir / "blank.pdf"
    _write_empty_pdf(pdf, n_pages=3)
    entry = CorpusManifestEntry(
        paper_id="paper_blank",
        title="Blank",
        year=2020,
        pdf_filename="blank.pdf",
        content_hash="0" * 16,
    )
    pipeline = IngestionPipeline(papers_dir=papers_dir, processed_dir=processed)
    result = pipeline.ingest_entry(entry)
    assert result.status == IngestionStatus.FAILED
    assert result.chunks == []
    assert not (processed / "chunks.jsonl").exists() or (
        JsonlRepository(processed / "chunks.jsonl", Chunk).count() == 0
    )


def test_pages_to_sections_keeps_provenance() -> None:
    intro = "We introduce the problem of retrieval for large language models. " * 15
    method = "Our method combines dense and sparse signals with fusion. " * 15
    pages = [
        PaperPage(
            paper_id="p",
            page_number=1,
            text=f"Abstract\nSummary of the paper about RAG systems and evaluation.\n\n1. Introduction\n{intro}",
            char_count=200,
        ),
        PaperPage(
            paper_id="p",
            page_number=2,
            text=f"2. Method\n{method}",
            char_count=200,
        ),
    ]
    sections = pages_to_sections(pages)
    assert sections
    assert sections[0].page_start == 1
    titles = [s.title for s in sections if s.title]
    assert any(t and "Introduction" in t for t in titles)
    assert any(t and "Method" in t for t in titles)

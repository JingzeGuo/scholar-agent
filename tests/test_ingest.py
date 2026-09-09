from __future__ import annotations

from pathlib import Path

import fitz

from scholar_agent.ingest import ingest_directory, ingest_pdf, split_page
from scholar_agent.models import load_chunks, save_chunks


def test_split_page_respects_size_and_overlap() -> None:
    text = " ".join(f"token-{index}" for index in range(200))
    chunks = split_page(text, max_chars=180, overlap=30)

    assert len(chunks) > 2
    assert all(len(chunk) <= 180 for chunk in chunks)
    assert any(
        set(left.split()).intersection(right.split())
        for left, right in zip(chunks, chunks[1:], strict=False)
    )


def test_pdf_chunks_keep_physical_pages(papers_dir: Path) -> None:
    chunks = ingest_pdf(papers_dir / "Self-RAG.pdf")

    assert {chunk["page"] for chunk in chunks} == {1, 2}
    assert all(chunk["paper"] == "Self-RAG.pdf" for chunk in chunks)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert [chunk["page_chunk_index"] for chunk in chunks if chunk["page"] == 1] == [0]
    assert [chunk["page_chunk_index"] for chunk in chunks if chunk["page"] == 2] == [0]
    assert "Self-RAG is" in next(chunk["text"] for chunk in chunks if chunk["page"] == 1)
    assert "controls retrieval" in next(chunk["text"] for chunk in chunks if chunk["page"] == 2)


def test_ingest_directory_writes_plain_jsonl(papers_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "processed" / "chunks.jsonl"
    written = ingest_directory(papers_dir, output)
    loaded = load_chunks(output)

    assert len(written) == len(loaded) == 4
    assert set(loaded[0]) == {
        "chunk_id",
        "paper",
        "page",
        "chunk_index",
        "page_chunk_index",
        "text",
        "score",
    }


def test_source_metadata_survives_ingestion_and_chunk_storage(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    with fitz.open() as document:
        document.set_metadata({"title": "Self-RAG: Learning to Retrieve"})
        document.new_page().insert_text((72, 72), "Self-RAG controls retrieval.")
        document.save(pdf_path)

    chunks = ingest_pdf(pdf_path)
    assert chunks[0]["title"] == "Self-RAG: Learning to Retrieve"
    assert "section" not in chunks[0]
    chunks[0]["section"] = "2. Method"
    output = tmp_path / "chunks.jsonl"
    save_chunks(chunks, output)
    loaded = load_chunks(output)

    assert loaded[0]["title"] == chunks[0]["title"]
    assert loaded[0]["section"] == "2. Method"
    assert loaded[0]["chunk_id"] == chunks[0]["chunk_id"]


def test_title_falls_back_to_first_page_heading(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "A Reliable Paper Title", fontsize=18)
        page.insert_text((72, 110), "Researcher Name", fontsize=10)
        page.insert_text((72, 150), "Abstract", fontsize=12)
        document.save(pdf_path)

    assert ingest_pdf(pdf_path)[0]["title"] == "A Reliable Paper Title"

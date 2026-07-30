from __future__ import annotations

from pathlib import Path

from scholar_agent.ingest import ingest_directory, ingest_pdf, split_page
from scholar_agent.models import load_chunks


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
    assert "Self-RAG is" in next(chunk["text"] for chunk in chunks if chunk["page"] == 1)
    assert "controls retrieval" in next(chunk["text"] for chunk in chunks if chunk["page"] == 2)


def test_ingest_directory_writes_plain_jsonl(papers_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "processed" / "chunks.jsonl"
    written = ingest_directory(papers_dir, output)
    loaded = load_chunks(output)

    assert len(written) == len(loaded) == 4
    assert set(loaded[0]) == {"chunk_id", "paper", "page", "text", "score"}

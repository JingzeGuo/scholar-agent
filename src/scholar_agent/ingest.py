"""Page-aware PDF ingestion with deliberately simple character chunking."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import fitz

from scholar_agent.models import save_chunks

LOGGER = logging.getLogger(__name__)
SPACE_RE = re.compile(r"\s+")
MIN_TITLE_SIZE = 13.0


def clean_text(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def _document_title(document: fitz.Document) -> str:
    title = clean_text(document.metadata.get("title") or "")
    if title and not (title.startswith("(") and title.endswith(")")):
        return title

    page = document[0]
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = clean_text("".join(span["text"] for span in line["spans"]))
            if (
                text
                and "arxiv:" not in text.casefold()
                and line["bbox"][1] < page.rect.height * 0.45
            ):
                lines.append(
                    (
                        line["bbox"][1],
                        line["bbox"][0],
                        max(span["size"] for span in line["spans"]),
                        text,
                    ),
                )
    title_size = max((size for _, _, size, _ in lines), default=0.0)
    if title_size < MIN_TITLE_SIZE:
        return ""
    return clean_text(
        " ".join(
            text
            for _, _, size, text in sorted(lines)
            if size >= title_size - 0.2
        ),
    )


def split_page(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Split one physical page; no returned chunk can cross a page boundary."""
    text = clean_text(text)
    if not text:
        return []
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Require max_chars > overlap >= 0")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, hard_end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        while start < end and text[start].isspace():
            start += 1
    return chunks


def _chunk_id(paper: str, page: int, position: int, text: str) -> str:
    identity = f"{paper}\0{page}\0{position}\0{text}".encode()
    return hashlib.sha1(identity).hexdigest()[:16]


def ingest_pdf(
    pdf_path: Path,
    *,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[dict]:
    """Extract and chunk one PDF while retaining its physical page number."""
    chunks: list[dict] = []
    with fitz.open(pdf_path) as document:
        title = _document_title(document)
        chunk_index = 0
        for page_number, page in enumerate(document, start=1):
            for page_chunk_index, text in enumerate(
                split_page(page.get_text("text"), max_chars, overlap),
            ):
                chunks.append(
                    {
                        "chunk_id": _chunk_id(pdf_path.name, page_number, page_chunk_index, text),
                        "paper": pdf_path.name,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "page_chunk_index": page_chunk_index,
                        "text": text,
                        **({"title": title} if title else {}),
                    },
                )
                chunk_index += 1
    return chunks


def ingest_directory(
    pdf_directory: Path,
    output_path: Path,
    *,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[dict]:
    """Ingest every PDF in a directory into one transparent JSONL file."""
    pdfs = sorted(pdf_directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {pdf_directory}")

    chunks: list[dict] = []
    for pdf_path in pdfs:
        paper_chunks = ingest_pdf(pdf_path, max_chars=max_chars, overlap=overlap)
        chunks.extend(paper_chunks)
        LOGGER.info("[ingest] %s pages produced %d chunks", pdf_path.name, len(paper_chunks))
    save_chunks(chunks, output_path)
    return chunks

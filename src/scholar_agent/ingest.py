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


def clean_text(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


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
        for page_number, page in enumerate(document, start=1):
            for position, text in enumerate(
                split_page(page.get_text("text"), max_chars, overlap),
            ):
                chunks.append(
                    {
                        "chunk_id": _chunk_id(pdf_path.name, page_number, position, text),
                        "paper": pdf_path.name,
                        "page": page_number,
                        "text": text,
                    },
                )
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

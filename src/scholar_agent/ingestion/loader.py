"""PDF loading with PyMuPDF, preserving page boundaries."""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from scholar_agent.ids import content_hash
from scholar_agent.models.corpus import PaperPage

# Heuristic: pages with almost no extractable text but non-trivial size may be scans
_MIN_CHARS_NONEMPTY = 40
_SCAN_MAX_CHARS = 80
_SCAN_MIN_IMAGES = 1


class PDFLoadError(ValueError):
    """Raised when a PDF cannot be opened or is unusable."""


def file_content_hash(path: Path) -> str:
    return content_hash(path.read_bytes(), length=16)


def open_pdf(path: Path | str) -> fitz.Document:
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise PDFLoadError(f"PDF not found: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 — surface as PDFLoadError
        raise PDFLoadError(f"failed to open PDF {pdf_path}: {exc}") from exc
    if doc.page_count < 1:
        doc.close()
        raise PDFLoadError(f"PDF has zero pages: {pdf_path}")
    return doc


def extract_page_text(page: fitz.Page) -> str:
    """Extract plain text from a page and normalize light whitespace."""
    # "text" preserves reading order better than "blocks" for body prose
    raw = page.get_text("text") or ""
    # Normalize newlines; keep paragraph breaks
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pages(paper_id: str, path: Path | str) -> tuple[list[PaperPage], dict[int, int]]:
    """Load all pages for a paper.

    Returns (pages, stats) where stats include image counts per page number.
    """
    pdf_path = Path(path)
    doc = open_pdf(pdf_path)
    pages: list[PaperPage] = []
    image_counts: dict[int, int] = {}
    try:
        for index in range(doc.page_count):
            page = doc.load_page(index)
            text = extract_page_text(page)
            page_number = index + 1  # 1-indexed provenance
            try:
                images = page.get_images(full=True) or []
            except Exception:  # noqa: BLE001
                images = []
            image_counts[page_number] = len(images)
            char_count = len(text)
            is_empty = char_count < _MIN_CHARS_NONEMPTY
            is_scanned = (
                char_count <= _SCAN_MAX_CHARS and image_counts[page_number] >= _SCAN_MIN_IMAGES
            ) or (char_count == 0 and image_counts[page_number] >= 1)
            pages.append(
                PaperPage(
                    paper_id=paper_id,
                    page_number=page_number,
                    text=text if not is_empty else text,
                    char_count=char_count,
                    is_empty=is_empty or not text.strip(),
                    is_scanned_suspect=is_scanned,
                )
            )
    finally:
        doc.close()
    return pages, image_counts


def extract_pdf_metadata(path: Path | str) -> dict[str, str | None]:
    """Best-effort PDF document metadata (title/author/etc.)."""
    doc = open_pdf(path)
    try:
        meta = doc.metadata or {}
        return {
            "title": (meta.get("title") or None),
            "author": (meta.get("author") or None),
            "subject": (meta.get("subject") or None),
            "creator": (meta.get("creator") or None),
            "producer": (meta.get("producer") or None),
        }
    finally:
        doc.close()

"""Merge corpus manifest metadata with PDF-level signals."""

from __future__ import annotations

from pathlib import Path

from scholar_agent.ingestion.loader import extract_pdf_metadata, file_content_hash
from scholar_agent.models.corpus import CorpusManifestEntry, Paper


def build_paper(
    entry: CorpusManifestEntry,
    pdf_path: Path,
    *,
    page_count: int | None = None,
    content_hash: str | None = None,
) -> Paper:
    """Construct a Paper record from manifest + file hash (+ optional PDF meta)."""
    digest = content_hash or file_content_hash(pdf_path)
    # Prefer curated manifest fields; PDF meta is fallback only
    pdf_meta = {}
    try:
        pdf_meta = extract_pdf_metadata(pdf_path)
    except Exception:  # noqa: BLE001 — metadata is best-effort
        pdf_meta = {}

    title = entry.title.strip() or (pdf_meta.get("title") or "Untitled").strip()
    authors = list(entry.authors)
    if not authors and pdf_meta.get("author"):
        authors = [a.strip() for a in str(pdf_meta["author"]).split(",") if a.strip()]

    return Paper(
        paper_id=entry.paper_id,
        title=title,
        authors=authors,
        year=entry.year,
        venue=entry.venue,
        arxiv_id=entry.arxiv_id,
        doi=entry.doi,
        source_url=entry.source_url,
        pdf_path=str(pdf_path),
        content_hash=digest,
        topic_labels=list(entry.topic_labels),
        page_count=page_count,
    )


def resolve_pdf_path(entry: CorpusManifestEntry, papers_dir: Path) -> Path:
    path = papers_dir / entry.pdf_filename
    if not path.is_file():
        # Allow absolute pdf_filename
        alt = Path(entry.pdf_filename)
        if alt.is_file():
            return alt
        raise FileNotFoundError(f"PDF missing for {entry.paper_id}: {path}")
    return path

"""Canonical provenance checks and PDF-page rendering for the demo."""

from __future__ import annotations

from pathlib import Path

import fitz

from scholar_agent.app.demo_models import DemoSessionResult, SavedDemoRun
from scholar_agent.ids import normalize_for_id
from scholar_agent.retrieval.chunk_store import ChunkStore

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_pdf_path(pdf_path: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a source-card path and reject files outside the repository."""
    raw = Path(pdf_path).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
    root = repo_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"PDF path escapes repository: {pdf_path}")
    if resolved.suffix.lower() != ".pdf":
        raise ValueError(f"source is not a PDF: {pdf_path}")
    return resolved


def render_pdf_page_png(
    pdf_path: str | Path,
    page_number: int,
    *,
    repo_root: Path = REPO_ROOT,
    zoom: float = 1.35,
) -> bytes:
    """Render one 1-based PDF page as PNG bytes for Streamlit."""
    path = resolve_pdf_path(pdf_path, repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    if page_number < 1:
        raise ValueError("page_number must be >= 1")
    with fitz.open(path) as document:
        if page_number > document.page_count:
            raise ValueError(f"page {page_number} exceeds PDF page count {document.page_count}")
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return bytes(pixmap.tobytes("png"))


def validate_session_provenance(
    session: DemoSessionResult,
    store: ChunkStore,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Validate claim -> evidence -> canonical chunk -> physical PDF page."""
    issues: list[str] = []
    cards = {card.evidence_id: card for card in session.source_cards}
    evidence = {item.evidence_id: item for item in session.evidence}

    for claim in session.claims:
        claim_id = str(claim.get("claim_id") or "unknown")
        for evidence_id in claim.get("evidence_ids") or []:
            if evidence_id not in cards:
                issues.append(f"claim {claim_id} references missing source card {evidence_id}")

    for evidence_id, card in cards.items():
        chunk = store.get_chunk(card.chunk_id)
        paper = store.get_paper(card.paper_id)
        if chunk is None:
            issues.append(f"source {evidence_id} has noncanonical chunk {card.chunk_id}")
            continue
        if chunk.paper_id != card.paper_id:
            issues.append(f"source {evidence_id} paper does not match canonical chunk")
        if card.page_start < chunk.page_start or card.page_end > chunk.page_end:
            issues.append(f"source {evidence_id} pages do not match canonical chunk")
        snippet = normalize_for_id(card.snippet)
        chunk_text = normalize_for_id(chunk.text)
        if snippet and snippet not in chunk_text:
            issues.append(f"source {evidence_id} snippet is not in canonical chunk")
        if paper is None:
            issues.append(f"source {evidence_id} has unknown paper {card.paper_id}")
            continue
        if paper.page_count is not None and card.page_end > paper.page_count:
            issues.append(f"source {evidence_id} exceeds canonical PDF page count")
        pdf_value = card.pdf_path or paper.pdf_path
        try:
            pdf = resolve_pdf_path(pdf_value, repo_root=repo_root)
        except ValueError as exc:
            issues.append(f"source {evidence_id}: {exc}")
        else:
            if not pdf.is_file():
                issues.append(f"source {evidence_id} PDF missing: {pdf}")

        item = evidence.get(evidence_id)
        if item is not None and (
            item.chunk_id != card.chunk_id
            or item.paper_id != card.paper_id
            or item.page_start != card.page_start
            or item.page_end != card.page_end
        ):
            issues.append(f"evidence {evidence_id} disagrees with its source card")

    report = session.final_answer.citation_report if session.final_answer else None
    if report is not None:
        for evidence_id in report.cited_evidence_ids:
            if evidence_id not in cards:
                issues.append(f"citation report references missing source {evidence_id}")
    return issues


def validate_saved_run_provenance(
    run: SavedDemoRun,
    store: ChunkStore,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    issues = validate_session_provenance(run.session, store, repo_root=repo_root)
    if run.corpus_fingerprint != store.fingerprint:
        issues.insert(
            0,
            "saved replay corpus fingerprint does not match canonical chunks",
        )
    if not run.provenance_verified:
        issues.insert(0, "saved replay is not marked provenance-verified")
    return issues

"""Extraction quality assessment and reporting."""

from __future__ import annotations

from scholar_agent.ingestion.tokens import count_tokens
from scholar_agent.models.corpus import PaperPage
from scholar_agent.models.ingestion import (
    CorpusIngestionReport,
    ExtractionIssue,
    ExtractionSeverity,
    PaperExtractionReport,
)


def assess_pages(
    paper_id: str,
    pdf_path: str,
    pages: list[PaperPage],
    *,
    chunk_count: int = 0,
) -> PaperExtractionReport:
    empty = sum(1 for p in pages if p.is_empty)
    scanned = sum(1 for p in pages if p.is_scanned_suspect)
    total_chars = sum(p.char_count for p in pages)
    full_text = "\n".join(p.text for p in pages if p.text)
    issues: list[ExtractionIssue] = []

    is_empty_paper = total_chars < 80 or all(p.is_empty for p in pages)
    is_scanned = (
        len(pages) > 0 and scanned / max(1, len(pages)) >= 0.5 and total_chars < 500 * len(pages)
    ) or (is_empty_paper and scanned > 0)

    if is_empty_paper:
        issues.append(
            ExtractionIssue(
                code="empty_paper",
                severity=ExtractionSeverity.ERROR,
                message="No extractable text; paper will not be indexed",
            )
        )
    if is_scanned:
        issues.append(
            ExtractionIssue(
                code="scanned_suspect",
                severity=ExtractionSeverity.WARNING,
                message=(
                    f"{scanned}/{len(pages)} pages look scanned/image-only; "
                    "OCR is out of scope for Phase 2"
                ),
            )
        )
    for page in pages:
        if page.is_empty:
            issues.append(
                ExtractionIssue(
                    code="empty_page",
                    severity=ExtractionSeverity.WARNING,
                    message="Page has little or no extractable text",
                    page_number=page.page_number,
                )
            )
        elif page.is_scanned_suspect:
            issues.append(
                ExtractionIssue(
                    code="scanned_page",
                    severity=ExtractionSeverity.INFO,
                    message="Page may be scanned (few chars + images)",
                    page_number=page.page_number,
                )
            )

    return PaperExtractionReport(
        paper_id=paper_id,
        pdf_path=pdf_path,
        page_count=len(pages),
        empty_page_count=empty,
        scanned_suspect_page_count=scanned,
        total_chars=total_chars,
        total_tokens_est=count_tokens(full_text),
        chunk_count=chunk_count,
        is_empty_paper=is_empty_paper,
        is_scanned_suspect=is_scanned,
        issues=issues,
    )


def summarize_report(report: CorpusIngestionReport) -> str:
    lines = [
        f"Ingestion run {report.run_id}",
        f"  attempted={report.papers_attempted} ingested={report.papers_ingested} "
        f"skipped={report.papers_skipped} failed={report.papers_failed}",
        f"  pages={report.total_pages} chunks={report.total_chunks}",
    ]
    if report.empty_papers:
        lines.append(f"  empty_papers={len(report.empty_papers)}")
    if report.scanned_suspect_papers:
        lines.append(f"  scanned_suspect={len(report.scanned_suspect_papers)}")
    for note in report.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)

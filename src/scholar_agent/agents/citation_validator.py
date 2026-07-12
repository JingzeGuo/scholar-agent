"""Citation validator (Phase 7).

Runs after the Writer draft. Ensures every citation ID exists, maps to a real
paper/chunk/page, and that cited evidence supports the attached claim. Removes
or explicitly qualifies unsupported claims before final output. Emits a
machine-readable CitationReport plus user-facing source cards / reference list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from scholar_agent.agents.writer import render_claim_markdown
from scholar_agent.ids import normalize_text
from scholar_agent.logging import get_logger
from scholar_agent.models.answer import (
    CitationIssue,
    CitationReport,
    ClaimWithCitations,
    DraftAnswer,
    FinalAnswer,
    SourceCard,
)
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.retrieval.chunk_store import ChunkStore

logger = get_logger(__name__)

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "also",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "supported",
    "the",
    "to",
    "with",
}


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", normalize_text(text))
        if t not in _STOP_WORDS and len(t) > 1
    }


def _snippet(text: str, max_chars: int = 200) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "…"


@dataclass
class CitationValidator:
    """Validate and repair draft answers against the evidence ledger."""

    min_support_overlap: float = 0.5
    require_per_claim_citations: bool = True
    provenance_store: ChunkStore | None = None
    require_pdf_provenance: bool = False
    _pdf_page_counts: dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def validate(
        self,
        draft: DraftAnswer,
        ledger: EvidenceLedger,
    ) -> FinalAnswer:
        """Return a FinalAnswer with citation report, source cards, and cleaned claims."""
        by_id = {item.evidence_id: item for item in ledger.items}
        issues: list[CitationIssue] = []
        final_claims: list[ClaimWithCitations] = []
        removed_qualifications: list[str] = []

        for claim in draft.claims:
            cleaned = self._validate_claim(claim, by_id, issues)
            if cleaned is not None and cleaned.evidence_ids:
                final_claims.append(cleaned)
                continue
            issues.append(
                CitationIssue(
                    severity="error",
                    claim_id=claim.claim_id,
                    message="Claim removed from primary answer: no valid citations",
                )
            )
            removed_qualifications.append(
                f"{claim.claim_id}: unsupported — removed from answer "
                f"(no valid supporting evidence ID). Original: {claim.text[:200]}"
            )

        # Paragraph-level citation anti-pattern: many claims but only some cited
        if self.require_per_claim_citations and len(draft.claims) > 1:
            uncited = [c for c in draft.claims if not c.evidence_ids]
            if uncited:
                issues.append(
                    CitationIssue(
                        severity="warning",
                        claim_id=uncited[0].claim_id,
                        message=(
                            f"{len(uncited)} claim(s) lacked per-claim citations "
                            "(paragraph-level citation anti-pattern)"
                        ),
                    )
                )

        cited_ids = self._ordered_cited_ids(final_claims, by_id)
        source_cards = [self._to_source_card(by_id[eid]) for eid in cited_ids]
        sources = [card.format_reference() for card in source_cards]

        # Validity: every remaining citation is real and supports its claim.
        is_valid = all(
            eid in by_id for c in final_claims for eid in c.evidence_ids
        ) and all(
            self._supports(c, by_id[eid])
            for c in final_claims
            for eid in c.evidence_ids
            if eid in by_id
        )
        # If we had to strip everything and draft claimed evidence, still report invalid
        if draft.claims and not final_claims and not draft.corpus_insufficient:
            is_valid = False
            if not any("no valid citations" in i.message for i in issues):
                issues.append(
                    CitationIssue(
                        severity="error",
                        message="All claims failed citation validation",
                    )
                )

        report = CitationReport(
            is_valid=is_valid,
            issues=issues,
            cited_evidence_ids=cited_ids,
            cited_paper_ids=sorted({by_id[e].paper_id for e in cited_ids}),
        )

        markdown = self._render_final_markdown(
            draft=draft,
            claims=final_claims,
            by_id=by_id,
            source_cards=source_cards,
            report=report,
            removed_qualifications=removed_qualifications,
        )

        final = FinalAnswer(
            markdown=markdown,
            claims=final_claims,
            sources=sources,
            source_cards=source_cards,
            citation_report=report,
            corpus_insufficient=draft.corpus_insufficient
            or (not final_claims and bool(draft.claims)),
        )
        logger.info(
            "citation_validated is_valid=%s claims=%s→%s issues=%s",
            report.is_valid,
            len(draft.claims),
            len(final_claims),
            len(issues),
        )
        return final

    def _validate_claim(
        self,
        claim: ClaimWithCitations,
        by_id: dict[str, EvidenceItem],
        issues: list[CitationIssue],
    ) -> ClaimWithCitations | None:
        if not claim.evidence_ids:
            issues.append(
                CitationIssue(
                    severity="error",
                    claim_id=claim.claim_id,
                    message="Claim has no evidence_ids",
                )
            )
            return None

        valid_ids: list[str] = []
        for eid in claim.evidence_ids:
            item = by_id.get(eid)
            if item is None:
                issues.append(
                    CitationIssue(
                        severity="error",
                        claim_id=claim.claim_id,
                        evidence_id=eid,
                        message=f"Citation refers to nonexistent evidence ID: {eid}",
                    )
                )
                continue
            # Provenance integrity
            if not item.paper_id or not item.chunk_id:
                issues.append(
                    CitationIssue(
                        severity="error",
                        claim_id=claim.claim_id,
                        evidence_id=eid,
                        message="Evidence missing paper_id or chunk_id",
                    )
                )
                continue
            if item.page_start < 1 or item.page_end < item.page_start:
                issues.append(
                    CitationIssue(
                        severity="error",
                        claim_id=claim.claim_id,
                        evidence_id=eid,
                        message="Evidence has invalid page range",
                    )
                )
                continue
            provenance_error = self._provenance_error(item)
            if provenance_error is not None:
                issues.append(
                    CitationIssue(
                        severity="error",
                        claim_id=claim.claim_id,
                        evidence_id=eid,
                        message=provenance_error,
                    )
                )
                continue
            if not self._supports(claim, item):
                issues.append(
                    CitationIssue(
                        severity="error",
                        claim_id=claim.claim_id,
                        evidence_id=eid,
                        message=(
                            "Cited evidence does not support the nearby claim "
                            f"(token overlap < {self.min_support_overlap})"
                        ),
                    )
                )
                continue
            if eid not in valid_ids:
                valid_ids.append(eid)

        if not valid_ids:
            return None
        return ClaimWithCitations(
            claim_id=claim.claim_id,
            text=claim.text,
            evidence_ids=valid_ids,
        )

    def _provenance_error(self, item: EvidenceItem) -> str | None:
        """Validate evidence against canonical chunk, paper, and physical PDF."""
        store = self.provenance_store
        if store is None:
            if self.require_pdf_provenance:
                return "Canonical provenance store unavailable"
            return None

        chunk = store.get_chunk(item.chunk_id)
        if chunk is None:
            return f"Canonical chunk not found: {item.chunk_id}"
        if chunk.paper_id != item.paper_id:
            return "Evidence paper_id does not match canonical chunk"
        if item.page_start < chunk.page_start or item.page_end > chunk.page_end:
            return "Evidence page range falls outside canonical chunk pages"
        if normalize_text(item.evidence_text) not in normalize_text(chunk.text):
            return "Evidence text does not map to the canonical chunk"

        paper = store.get_paper(item.paper_id)
        if paper is None:
            return f"Canonical paper not found: {item.paper_id}"
        pdf_path = self._resolve_pdf_path(paper.pdf_path, store)
        if not pdf_path.is_file():
            return f"Source PDF not found: {pdf_path}"
        try:
            actual_pages = self._pdf_page_count(pdf_path)
        except (OSError, RuntimeError, ValueError) as exc:
            return f"Source file is not a readable PDF: {type(exc).__name__}"
        if item.page_end > actual_pages:
            return (
                f"Evidence page {item.page_end} exceeds PDF page count "
                f"{actual_pages}"
            )
        if paper.page_count is not None and item.page_end > paper.page_count:
            return "Evidence page exceeds canonical paper page_count"
        return None

    def _resolve_pdf_path(self, value: str, store: ChunkStore) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        cwd_candidate = (Path.cwd() / path).resolve()
        if cwd_candidate.is_file() or store.papers_path is None:
            return cwd_candidate
        # A portable processed store may use paths relative to its repository root.
        roots = list(store.papers_path.resolve().parents)
        for root in roots:
            candidate = (root / path).resolve()
            if candidate.is_file():
                return candidate
        return cwd_candidate

    def _pdf_page_count(self, path: Path) -> int:
        key = str(path.resolve())
        cached = self._pdf_page_counts.get(key)
        if cached is not None:
            return cached
        with pymupdf.open(path) as document:
            if not document.is_pdf:
                raise ValueError("not a PDF")
            count = int(document.page_count)
        if count < 1:
            raise ValueError("PDF has no pages")
        self._pdf_page_counts[key] = count
        return count

    def _supports(self, claim: ClaimWithCitations, item: EvidenceItem) -> bool:
        """Heuristic entailment: claim tokens should appear in evidence text/claim."""
        # Meta / qualification claims about conflicts are allowed if they cite the items
        meta_markers = (
            "sources disagree",
            "conflicting passages",
            "retained for audit",
            "limitation",
        )
        claim_l = claim.text.lower()
        if any(m in claim_l for m in meta_markers):
            return True

        claim_toks = _tokens(claim.text)
        if not claim_toks:
            return True
        evidence_blob = f"{item.claim} {item.evidence_text}"
        ev_toks = _tokens(evidence_blob)
        if not ev_toks:
            return False
        claim_numbers = {token for token in claim_toks if token.isdigit()}
        if not claim_numbers.issubset(ev_toks):
            return False
        polarity_roots = (
            ("outperform", "underperform"),
            ("increase", "decrease"),
            ("better", "worse"),
            ("effective", "ineffective"),
        )
        for positive, negative in polarity_roots:
            claim_positive = any(token.startswith(positive) for token in claim_toks)
            claim_negative = any(token.startswith(negative) for token in claim_toks)
            evidence_positive = any(token.startswith(positive) for token in ev_toks)
            evidence_negative = any(token.startswith(negative) for token in ev_toks)
            if (claim_positive and evidence_negative) or (
                claim_negative and evidence_positive
            ):
                return False
        # Also allow if claim is largely a substring of evidence (after normalize)
        if normalize_text(claim.text)[:80] in normalize_text(evidence_blob):
            return True
        overlap = len(claim_toks & ev_toks) / max(1, len(claim_toks))
        return overlap >= self.min_support_overlap

    def _ordered_cited_ids(
        self,
        claims: list[ClaimWithCitations],
        by_id: dict[str, EvidenceItem],
    ) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for claim in claims:
            for eid in claim.evidence_ids:
                if eid in by_id and eid not in seen:
                    ordered.append(eid)
                    seen.add(eid)
        return ordered

    def _to_source_card(self, item: EvidenceItem) -> SourceCard:
        paper = (
            self.provenance_store.get_paper(item.paper_id)
            if self.provenance_store is not None
            else None
        )
        pdf_path = None
        if paper is not None and self.provenance_store is not None:
            pdf_path = str(
                self._resolve_pdf_path(paper.pdf_path, self.provenance_store)
            )
        return SourceCard(
            evidence_id=item.evidence_id,
            paper_id=item.paper_id,
            chunk_id=item.chunk_id,
            page_start=item.page_start,
            page_end=item.page_end,
            snippet=_snippet(item.evidence_text),
            retrieval_method=item.retrieval_method,
            title=paper.title if paper else None,
            pdf_path=pdf_path,
        )

    def _render_final_markdown(
        self,
        *,
        draft: DraftAnswer,
        claims: list[ClaimWithCitations],
        by_id: dict[str, EvidenceItem],
        source_cards: list[SourceCard],
        report: CitationReport,
        removed_qualifications: list[str] | None = None,
    ) -> str:
        # Prefer re-render from cleaned claims so dropped citations never appear
        lines: list[str] = [
            "## Answer",
            "",
        ]
        # Preserve question line if present in draft
        for line in draft.markdown.splitlines():
            if line.startswith("**Question:**"):
                lines.append(line)
                lines.append("")
                break

        if draft.corpus_insufficient or not claims or removed_qualifications:
            lines.extend(
                [
                    "> **Limitation:** The answer is restricted to verified evidence. "
                    "Unsupported claims were removed or the corpus is insufficient.",
                    "",
                ]
            )

        lines.append("### Claims")
        lines.append("")
        if not claims:
            lines.append("No citation-validated claims remain.")
        else:
            for claim in claims:
                lines.append(f"- {render_claim_markdown(claim, by_id)}")
        lines.append("")

        notes = list(draft.notes)
        if removed_qualifications:
            notes.extend(removed_qualifications)
        if notes:
            lines.append("### Notes")
            lines.append("")
            for note in notes:
                lines.append(f"- {note}")
            lines.append("")

        if source_cards:
            lines.append("### Sources")
            lines.append("")
            for card in source_cards:
                title = f" — {card.title}" if card.title else ""
                lines.append(
                    f"- {card.format_inline()}{title} · `{card.chunk_id}` · "
                    f"`{card.evidence_id}`"
                )
                if card.pdf_path:
                    lines.append(f"  - PDF: `{card.pdf_path}`")
                if card.snippet:
                    lines.append(f"  - {card.snippet}")
            lines.append("")

        lines.append("### Citation validation")
        lines.append("")
        lines.append(
            f"- valid={report.is_valid} · cited_evidence={len(report.cited_evidence_ids)} "
            f"· issues={len(report.issues)}"
        )
        for issue in report.issues[:12]:
            loc = issue.claim_id or issue.evidence_id or "—"
            lines.append(f"- [{issue.severity}] {loc}: {issue.message}")
        lines.append("")

        return "\n".join(lines).rstrip() + "\n"

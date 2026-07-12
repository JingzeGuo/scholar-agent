"""Citation validator (Phase 7).

Runs after the Writer draft. Ensures every citation ID exists, maps to a real
paper/chunk/page, and that cited evidence supports the attached claim. Removes
or explicitly qualifies unsupported claims before final output. Emits a
machine-readable CitationReport plus user-facing source cards / reference list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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

    min_support_overlap: float = 0.12
    require_per_claim_citations: bool = True

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
        return SourceCard(
            evidence_id=item.evidence_id,
            paper_id=item.paper_id,
            chunk_id=item.chunk_id,
            page_start=item.page_start,
            page_end=item.page_end,
            snippet=_snippet(item.evidence_text),
            retrieval_method=item.retrieval_method,
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
                lines.append(
                    f"- {card.format_inline()} · `{card.chunk_id}` · "
                    f"`{card.evidence_id}`"
                )
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

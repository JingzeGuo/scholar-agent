"""Evidence-constrained Writer (Phase 7).

Reads only the question, answer format, verified evidence ledger, contradiction
notes, and corpus-insufficiency notes. Does not retrieve. Emits structured
claims with evidence IDs first, then renders Markdown with inline citations
derived from those IDs.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from scholar_agent.ids import normalize_text
from scholar_agent.logging import get_logger
from scholar_agent.models.answer import ClaimWithCitations, DraftAnswer
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.models.planning import QueryPlan
from scholar_agent.models.workflow import VerificationResult

logger = get_logger(__name__)

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _tokens(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9]+", normalize_text(text))
        if t not in _STOP_WORDS and len(t) > 1
    ]


def _snippet(text: str, max_chars: int = 280) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "…"


def format_inline_citation(item: EvidenceItem) -> str:
    """User-facing page citation from evidence provenance."""
    if item.page_start == item.page_end:
        return f"[{item.paper_id} p.{item.page_start}]"
    return f"[{item.paper_id} p.{item.page_start}-{item.page_end}]"


def render_claim_markdown(
    claim: ClaimWithCitations,
    ledger_by_id: dict[str, EvidenceItem],
) -> str:
    """Render one claim with inline citations derived from evidence IDs."""
    markers: list[str] = []
    seen: set[str] = set()
    for eid in claim.evidence_ids:
        item = ledger_by_id.get(eid)
        if item is None:
            continue
        marker = format_inline_citation(item)
        if marker not in seen:
            markers.append(marker)
            seen.add(marker)
    if markers:
        return f"{claim.text} {' '.join(markers)}"
    return claim.text


@dataclass
class Writer:
    """Deterministic offline Writer (LLM path optional later)."""

    max_claims: int = 12
    max_evidence_per_claim: int = 3

    def write(
        self,
        *,
        query: str,
        plan: QueryPlan | None = None,
        ledger: EvidenceLedger,
        verification: VerificationResult | None = None,
        answer_format: str | None = None,
        contradiction_notes: list[str] | None = None,
        corpus_insufficient: bool | None = None,
    ) -> DraftAnswer:
        """Produce claim→evidence draft then Markdown.

        Only ledger items and provided notes may support content. Gaps are
        stated explicitly rather than filled from model memory.
        """
        notes: list[str] = []
        if contradiction_notes:
            notes.extend(contradiction_notes)
        elif verification and verification.conflicting_evidence_ids:
            notes.append(
                "Conflicting evidence retained: "
                + ", ".join(verification.conflicting_evidence_ids[:12])
            )
        if verification and verification.missing_aspects:
            notes.append(
                "Missing aspects after verification: "
                + "; ".join(verification.missing_aspects[:8])
            )

        insufficient = (
            bool(corpus_insufficient)
            if corpus_insufficient is not None
            else bool(verification and verification.unanswerable)
        )
        if insufficient:
            notes.append(
                "Corpus is insufficient to fully answer the question; "
                "claims below are limited to available evidence."
            )

        claims = self._build_claims(query=query, plan=plan, ledger=ledger)
        if not claims and not ledger.items:
            insufficient = True
            notes.append("No verified evidence was available for writing.")
            draft = DraftAnswer(
                claims=[],
                markdown=self._render_empty(query, notes),
                corpus_insufficient=True,
                notes=notes,
            )
            logger.info("writer_empty query_len=%s", len(query))
            return draft

        fmt = answer_format or (plan.answer_type if plan else "factual")
        markdown = self._render_markdown(
            query=query,
            answer_format=fmt,
            claims=claims,
            ledger=ledger,
            notes=notes,
            corpus_insufficient=insufficient,
            verification=verification,
        )
        draft = DraftAnswer(
            claims=claims,
            markdown=markdown,
            corpus_insufficient=insufficient,
            notes=notes,
        )
        logger.info(
            "writer_draft claims=%s corpus_insufficient=%s",
            len(claims),
            insufficient,
        )
        return draft

    def _build_claims(
        self,
        *,
        query: str,
        plan: QueryPlan | None,
        ledger: EvidenceLedger,
    ) -> list[ClaimWithCitations]:
        if not ledger.items:
            return []

        by_sq: dict[str, list[EvidenceItem]] = defaultdict(list)
        for item in ledger.items:
            by_sq[item.sub_question_id].append(item)

        # Preserve plan order when available, then any remaining sub-questions
        ordered_sq_ids: list[str] = []
        if plan is not None:
            for sq in plan.sub_questions:
                if sq.id in by_sq:
                    ordered_sq_ids.append(sq.id)
        for sq_id in by_sq:
            if sq_id not in ordered_sq_ids:
                ordered_sq_ids.append(sq_id)

        claims: list[ClaimWithCitations] = []
        claim_idx = 0
        for sq_id in ordered_sq_ids:
            items = sorted(
                by_sq[sq_id],
                key=lambda e: (
                    e.support_score if e.support_score is not None else -1.0,
                    e.rerank_score if e.rerank_score is not None else -1.0,
                    e.retrieval_score if e.retrieval_score is not None else -1.0,
                ),
                reverse=True,
            )
            # Prefer non-contradiction items for primary claims; still surface conflicts later
            primary = [i for i in items if not i.contradiction] or list(items)
            # Group by paper for diversity within the sub-question
            selected = self._select_diverse(primary, limit=self.max_evidence_per_claim)
            if not selected:
                continue
            claim_idx += 1
            text = self._claim_text_from_evidence(selected, query=query)
            claims.append(
                ClaimWithCitations(
                    claim_id=f"claim_{claim_idx}",
                    text=text,
                    evidence_ids=[e.evidence_id for e in selected],
                )
            )
            if len(claims) >= self.max_claims:
                break

        # Surface contradiction notes as qualified claims when present
        contradicted = [i for i in ledger.items if i.contradiction]
        if contradicted and len(claims) < self.max_claims:
            claim_idx += 1
            sample = contradicted[: self.max_evidence_per_claim]
            claims.append(
                ClaimWithCitations(
                    claim_id=f"claim_{claim_idx}",
                    text=(
                        "Sources disagree on some details; conflicting passages "
                        "are retained for audit rather than resolved by the Writer."
                    ),
                    evidence_ids=[e.evidence_id for e in sample],
                )
            )
        return claims

    def _select_diverse(
        self, items: list[EvidenceItem], *, limit: int
    ) -> list[EvidenceItem]:
        if limit <= 0 or not items:
            return []
        selected: list[EvidenceItem] = []
        seen_papers: set[str] = set()
        # First pass: one per paper
        for item in items:
            if item.paper_id in seen_papers:
                continue
            selected.append(item)
            seen_papers.add(item.paper_id)
            if len(selected) >= limit:
                return selected
        # Second pass: fill remaining slots
        for item in items:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def _claim_text_from_evidence(
        self, items: list[EvidenceItem], *, query: str
    ) -> str:
        """Ground claim text in evidence text / evidence claim fields only."""
        # Prefer the best item's claim field if it is non-generic; else synthesize from snippet
        best = items[0]
        claim = best.claim.strip()
        # If claim is just a copy of the query or empty-ish, use evidence snippet
        q_norm = normalize_text(query)
        if not claim or normalize_text(claim) == q_norm or len(claim) < 12:
            claim = _snippet(best.evidence_text, 220)
        else:
            # Ensure claim does not invent content beyond evidence: require some token overlap
            claim_toks = set(_tokens(claim))
            ev_toks = set(_tokens(best.evidence_text))
            if claim_toks and ev_toks:
                overlap = len(claim_toks & ev_toks) / max(1, len(claim_toks))
                if overlap < 0.15:
                    claim = _snippet(best.evidence_text, 220)
        # Multi-source: lightly note additional papers without inventing synthesis
        extra_papers = sorted({i.paper_id for i in items[1:] if i.paper_id != best.paper_id})
        if extra_papers:
            claim = f"{claim.rstrip('.')} (also supported by {', '.join(extra_papers[:3])})."
        return claim

    def _render_markdown(
        self,
        *,
        query: str,
        answer_format: str,
        claims: list[ClaimWithCitations],
        ledger: EvidenceLedger,
        notes: list[str],
        corpus_insufficient: bool,
        verification: VerificationResult | None,
    ) -> str:
        by_id = {e.evidence_id: e for e in ledger.items}
        lines: list[str] = [
            "## Answer",
            "",
            f"**Question:** {query}",
            "",
            f"**Format:** {answer_format}",
            "",
        ]
        if corpus_insufficient:
            lines.extend(
                [
                    "> **Limitation:** The corpus does not fully support a complete "
                    "answer. Only evidence-backed claims are listed below.",
                    "",
                ]
            )

        lines.append("### Claims")
        lines.append("")
        if not claims:
            lines.append(
                "No evidence-backed claims could be formed from the verified ledger."
            )
        else:
            for claim in claims:
                rendered = render_claim_markdown(claim, by_id)
                lines.append(f"- {rendered}")
        lines.append("")

        if notes:
            lines.append("### Notes")
            lines.append("")
            for note in notes:
                lines.append(f"- {note}")
            lines.append("")

        if verification and verification.conflicting_evidence_ids:
            lines.append("### Contradictions")
            lines.append("")
            lines.append(
                "The following evidence IDs were flagged as potentially conflicting "
                "and are retained rather than suppressed:"
            )
            lines.append(
                "- " + ", ".join(verification.conflicting_evidence_ids[:12])
            )
            lines.append("")

        # Reference section built from cited evidence IDs in claim order
        cited_ids: list[str] = []
        seen: set[str] = set()
        for claim in claims:
            for eid in claim.evidence_ids:
                if eid not in seen and eid in by_id:
                    cited_ids.append(eid)
                    seen.add(eid)
        if cited_ids:
            lines.append("### References")
            lines.append("")
            for eid in cited_ids:
                item = by_id[eid]
                lines.append(
                    f"- {format_inline_citation(item)} "
                    f"`{item.chunk_id}` · `{item.evidence_id}`"
                )
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _render_empty(self, query: str, notes: list[str]) -> str:
        lines = [
            "## Answer",
            "",
            f"**Question:** {query}",
            "",
            "> **Limitation:** No verified evidence was available. The system "
            "cannot answer from the corpus without inventing content.",
            "",
        ]
        if notes:
            lines.append("### Notes")
            lines.append("")
            for note in notes:
                lines.append(f"- {note}")
            lines.append("")
        return "\n".join(lines)

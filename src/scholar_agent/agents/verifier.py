"""Independent Verification Agent.

Sees the query, plan, and evidence ledger only — not Research Agent hidden
reasoning. Emits structured VerificationResult with concrete corrective queries
when evidence is insufficient. Offline-deterministic by default.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from scholar_agent.ids import normalize_text
from scholar_agent.logging import get_logger
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.models.planning import QueryPlan, SubQuestion, SubQuestionStatus
from scholar_agent.models.workflow import VerificationResult

logger = get_logger(__name__)

# Lightweight contradiction cues
_NEGATION_PAIRS = [
    ("outperform", "underperform"),
    ("improve", "degrade"),
    ("better", "worse"),
    ("increases", "decreases"),
    ("superior", "inferior"),
    ("effective", "ineffective"),
    ("succeeds", "fails"),
]


@dataclass
class Verifier:
    """Coverage / relevance / contradiction / diversity checks."""

    min_evidence_per_sub_question: int = 1
    min_relevance_token_overlap: float = 0.08
    min_source_diversity: int = 2

    def verify(
        self,
        *,
        query: str,
        plan: QueryPlan,
        ledger: EvidenceLedger,
    ) -> VerificationResult:
        by_sq: dict[str, list[EvidenceItem]] = defaultdict(list)
        for item in ledger.items:
            by_sq[item.sub_question_id].append(item)

        covered: list[str] = []
        missing: list[str] = []
        missing_aspects: list[str] = []
        unsupported: list[str] = []
        corrective: list[str] = []

        for sq in plan.sub_questions:
            items = by_sq.get(sq.id, [])
            relevant = [e for e in items if self._is_relevant(sq, e)]
            if len(relevant) >= self.min_evidence_per_sub_question:
                covered.append(sq.id)
                # Check required evidence keywords lightly
                for req in sq.required_evidence:
                    if not self._requirement_covered(req, relevant):
                        missing_aspects.append(f"{sq.id}: missing aspect '{req}'")
                        corrective.append(self._corrective_for_aspect(sq, req))
            else:
                missing.append(sq.id)
                if not items:
                    missing_aspects.append(f"{sq.id}: no evidence retrieved")
                    corrective.append(
                        f"Find supporting passages for: {sq.question}"
                    )
                else:
                    missing_aspects.append(
                        f"{sq.id}: evidence not sufficiently relevant "
                        f"({len(relevant)}/{len(items)} relevant)"
                    )
                    corrective.append(
                        f"Retrieve more relevant evidence for: {sq.question}"
                    )
                if items:
                    # Mark weak claims as unsupported
                    for e in items:
                        if e not in relevant:
                            unsupported.append(e.claim[:120])

        conflicts = self._find_conflicts(ledger.items)
        diversity_ok, diversity_note = self._check_diversity(plan, ledger)

        n = max(1, len(plan.sub_questions))
        coverage_score = len(covered) / n
        # Penalize missing aspects and diversity failures
        if missing_aspects:
            coverage_score *= max(0.0, 1.0 - 0.1 * len(missing_aspects))
        if not diversity_ok:
            coverage_score = min(coverage_score, 0.7)
            missing_aspects.append(diversity_note)
            if not any("diverse" in c.lower() for c in corrective):
                corrective.append(
                    f"Gather evidence from additional distinct papers for: {query}"
                )

        unanswerable = self._detect_unanswerable(plan, ledger, covered, missing)
        is_sufficient = (
            not missing
            and not unanswerable
            and coverage_score >= 0.99
            and diversity_ok
            and not missing_aspects
        )
        # Soft sufficiency: all sub-questions covered, diversity ok, no aspect gaps
        if (
            not missing
            and diversity_ok
            and not unanswerable
            and coverage_score >= 0.85
            and len(missing_aspects) == 0
        ):
            is_sufficient = True

        if unanswerable:
            is_sufficient = False
            corrective = []  # do not thrash retrieval if corpus cannot answer
            missing_aspects.append("corpus_cannot_answer")

        rationale = self._rationale(
            is_sufficient=is_sufficient,
            covered=covered,
            missing=missing,
            conflicts=conflicts,
            diversity_ok=diversity_ok,
            unanswerable=unanswerable,
            coverage_score=coverage_score,
        )

        # Deduplicate corrective queries while preserving order
        seen_c: set[str] = set()
        unique_corrective: list[str] = []
        for c in corrective:
            key = normalize_text(c)
            if key not in seen_c:
                seen_c.add(key)
                unique_corrective.append(c)

        result = VerificationResult(
            is_sufficient=is_sufficient,
            coverage_score=round(min(1.0, max(0.0, coverage_score)), 3),
            covered_sub_questions=covered,
            missing_sub_questions=missing,
            unsupported_claims=unsupported[:20],
            conflicting_evidence_ids=conflicts,
            missing_aspects=missing_aspects[:20],
            corrective_queries=unique_corrective[:10],
            rationale_summary=rationale,
        )
        logger.info(
            "verification sufficient=%s coverage=%.2f missing=%s conflicts=%s",
            result.is_sufficient,
            result.coverage_score,
            len(result.missing_sub_questions),
            len(result.conflicting_evidence_ids),
        )
        return result

    def update_sub_question_status(
        self, plan: QueryPlan, verification: VerificationResult
    ) -> QueryPlan:
        """Return a copy of the plan with sub-question statuses updated."""
        missing = set(verification.missing_sub_questions)
        covered = set(verification.covered_sub_questions)
        updated: list[SubQuestion] = []
        for sq in plan.sub_questions:
            if sq.id in covered and sq.id not in missing:
                status = SubQuestionStatus.COVERED
            elif sq.id in missing:
                status = SubQuestionStatus.MISSING
            else:
                status = sq.status
            updated.append(sq.model_copy(update={"status": status}))
        return plan.model_copy(update={"sub_questions": updated})

    def _is_relevant(self, sq: SubQuestion, item: EvidenceItem) -> bool:
        q_tokens = set(re.findall(r"[a-z0-9]+", normalize_text(sq.question)))
        e_tokens = set(re.findall(r"[a-z0-9]+", normalize_text(item.evidence_text)))
        if not q_tokens or not e_tokens:
            return False
        overlap = len(q_tokens & e_tokens) / len(q_tokens)
        if overlap >= self.min_relevance_token_overlap:
            return True
        # Also accept if any required evidence keyword appears
        for req in sq.required_evidence:
            req_t = set(re.findall(r"[a-z0-9]+", normalize_text(req)))
            if req_t and len(req_t & e_tokens) / len(req_t) >= 0.5:
                return True
        return False

    def _requirement_covered(self, req: str, items: list[EvidenceItem]) -> bool:
        req_norm = normalize_text(req)
        # Soft requirements like "supporting passage" always ok if any item exists
        if req_norm in {
            "supporting passage",
            "definition_or_fact",
            "definition",
            "passages",
            "both sides",
            "comparison",
        }:
            return bool(items)
        req_tokens = set(re.findall(r"[a-z0-9]+", req_norm))
        if not req_tokens:
            return bool(items)
        for item in items:
            e_tokens = set(re.findall(r"[a-z0-9]+", normalize_text(item.evidence_text)))
            if len(req_tokens & e_tokens) / len(req_tokens) >= 0.4:
                return True
        return False

    def _corrective_for_aspect(self, sq: SubQuestion, req: str) -> str:
        return f"Find evidence about '{req}' for sub-question: {sq.question}"

    def _find_conflicts(self, items: list[EvidenceItem]) -> list[str]:
        """Detect simple cross-source polarity conflicts; retain both IDs."""
        if len(items) < 2:
            return []
        # Group by paper for cross-source checks
        by_paper: dict[str, list[EvidenceItem]] = defaultdict(list)
        for item in items:
            by_paper[item.paper_id].append(item)

        conflict_ids: list[str] = []
        texts = [(i.evidence_id, normalize_text(i.evidence_text), i.paper_id) for i in items]
        for i, (id_a, text_a, paper_a) in enumerate(texts):
            for id_b, text_b, paper_b in texts[i + 1 :]:
                if paper_a == paper_b:
                    continue
                for pos, neg in _NEGATION_PAIRS:
                    if (pos in text_a and neg in text_b) or (neg in text_a and pos in text_b):
                        conflict_ids.extend([id_a, id_b])
        # Also surface explicit contradiction flags on items
        for item in items:
            if item.contradiction:
                conflict_ids.append(item.evidence_id)

        # unique preserve order
        seen: set[str] = set()
        out: list[str] = []
        for eid in conflict_ids:
            if eid not in seen:
                seen.add(eid)
                out.append(eid)
        return out

    def _check_diversity(
        self, plan: QueryPlan, ledger: EvidenceLedger
    ) -> tuple[bool, str]:
        papers = {e.paper_id for e in ledger.items}
        need = max(plan.expected_source_diversity, self.min_source_diversity)
        # Only enforce multi-source diversity for comparison/synthesis
        if plan.answer_type not in {"comparison", "synthesis"}:
            return True, ""
        if len(papers) >= min(need, 2):
            return True, ""
        return (
            False,
            f"source diversity too low ({len(papers)} papers; expected ≥{min(need, 2)})",
        )

    def _detect_unanswerable(
        self,
        plan: QueryPlan,
        ledger: EvidenceLedger,
        covered: list[str],
        missing: list[str],
    ) -> bool:
        """Heuristic: empty ledger after research, or zero relevance across all items."""
        if not plan.sub_questions:
            return True
        if not ledger.items and missing:
            return True
        # All sub-questions missing and every item fails basic token overlap with original query
        if missing and len(covered) == 0 and ledger.items:
            q_tokens = set(re.findall(r"[a-z0-9]+", normalize_text(plan.original_query)))
            any_overlap = False
            for item in ledger.items:
                e_tokens = set(re.findall(r"[a-z0-9]+", normalize_text(item.evidence_text)))
                if q_tokens and len(q_tokens & e_tokens) / len(q_tokens) >= 0.05:
                    any_overlap = True
                    break
            if not any_overlap:
                return True
        return False

    def _rationale(
        self,
        *,
        is_sufficient: bool,
        covered: list[str],
        missing: list[str],
        conflicts: list[str],
        diversity_ok: bool,
        unanswerable: bool,
        coverage_score: float,
    ) -> str:
        if unanswerable:
            return (
                f"Corpus appears unable to answer (coverage={coverage_score:.2f}; "
                f"covered={len(covered)} missing={len(missing)})."
            )
        if is_sufficient:
            extra = ""
            if conflicts:
                extra = f" Conflicting evidence retained ({len(conflicts)} ids)."
            return (
                f"Evidence sufficient (coverage={coverage_score:.2f}; "
                f"covered={len(covered)}).{extra}"
            )
        parts = [
            f"Evidence insufficient (coverage={coverage_score:.2f}",
            f"covered={len(covered)}",
            f"missing={len(missing)})",
        ]
        if not diversity_ok:
            parts.append("source diversity low")
        if conflicts:
            parts.append(f"conflicts={len(conflicts)}")
        return "; ".join(parts) + "."

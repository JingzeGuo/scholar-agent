"""Deterministic citation quality metrics."""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from scholar_agent.evaluation.dataset import EvalQuestion
from scholar_agent.models.answer import FinalAnswer
from scholar_agent.models.retrieval import CitationRef, NaiveRAGAnswer


class CitationMetrics(BaseModel):
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_validity_rate: float = 0.0
    page_traceability_rate: float = 0.0
    unsupported_claim_rate: float = 0.0
    n_cited: int = 0
    n_claims: int = 0


class CitationMetricAggregate(BaseModel):
    n: int = 0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_validity_rate: float = 0.0
    page_traceability_rate: float = 0.0
    unsupported_claim_rate: float = 0.0
    by_type: dict[str, dict[str, float]] = Field(default_factory=dict)


_PAPER_PAGE_RE = re.compile(
    r"\[(?P<paper>paper_[a-z0-9_]+)\s+p\.(?P<start>\d+)(?:-(?P<end>\d+))?\]",
    re.IGNORECASE,
)


def _papers_from_naive(answer: NaiveRAGAnswer) -> set[str]:
    return {c.paper_id for c in answer.citations}


def _papers_from_final(answer: FinalAnswer) -> set[str]:
    papers = {c.paper_id for c in answer.source_cards}
    # Also parse inline markers from markdown
    for match in _PAPER_PAGE_RE.finditer(answer.markdown or ""):
        papers.add(match.group("paper"))
    return papers


def _page_traceable_final(answer: FinalAnswer) -> tuple[int, int]:
    cards = answer.source_cards
    if not cards:
        # parse markdown citations
        matches = list(_PAPER_PAGE_RE.finditer(answer.markdown or ""))
        ok = sum(1 for m in matches if int(m.group("start")) >= 1)
        return ok, len(matches)
    ok = sum(1 for c in cards if c.page_start >= 1 and c.page_end >= c.page_start)
    return ok, len(cards)


def _page_traceable_naive(answer: NaiveRAGAnswer) -> tuple[int, int]:
    if not answer.citations:
        return 0, 0
    ok = sum(1 for c in answer.citations if c.page_start >= 1 and c.page_end >= c.page_start)
    return ok, len(answer.citations)


def compute_citation_metrics_from_papers(
    question: EvalQuestion,
    cited_papers: set[str],
    *,
    validity_rate: float = 1.0,
    page_ok: int = 0,
    page_total: int = 0,
    n_claims: int = 0,
    n_unsupported_claims: int = 0,
) -> CitationMetrics:
    gold = question.gold_paper_ids()
    if question.unanswerable:
        # Prefer no citations; citing is OK only if validity still holds
        precision = 1.0 if not cited_papers else 0.0
        recall = 1.0  # nothing required
        return CitationMetrics(
            citation_precision=precision,
            citation_recall=recall,
            citation_validity_rate=validity_rate if cited_papers else 1.0,
            page_traceability_rate=(page_ok / page_total) if page_total else 1.0,
            unsupported_claim_rate=(n_unsupported_claims / n_claims if n_claims else 0.0),
            n_cited=len(cited_papers),
            n_claims=n_claims,
        )

    if not cited_papers:
        return CitationMetrics(
            citation_precision=0.0,
            citation_recall=0.0 if gold else 1.0,
            citation_validity_rate=1.0,
            page_traceability_rate=1.0 if page_total == 0 else 0.0,
            unsupported_claim_rate=1.0 if n_claims else 0.0,
            n_cited=0,
            n_claims=n_claims,
        )

    if gold:
        tp = cited_papers & gold
        precision = len(tp) / len(cited_papers)
        recall = len(tp) / len(gold)
    else:
        precision = 1.0
        recall = 1.0

    return CitationMetrics(
        citation_precision=precision,
        citation_recall=recall,
        citation_validity_rate=validity_rate,
        page_traceability_rate=(page_ok / page_total) if page_total else 0.0,
        unsupported_claim_rate=(n_unsupported_claims / n_claims if n_claims else 0.0),
        n_cited=len(cited_papers),
        n_claims=n_claims,
    )


def compute_citation_metrics_naive(
    question: EvalQuestion, answer: NaiveRAGAnswer
) -> CitationMetrics:
    papers = _papers_from_naive(answer)
    page_ok, page_total = _page_traceable_naive(answer)
    # Naive answers don't have claim objects; treat missing citations as unsupported body
    n_claims = 1 if answer.answer.strip() else 0
    n_unsupported = 0 if papers or question.unanswerable else n_claims
    return compute_citation_metrics_from_papers(
        question,
        papers,
        validity_rate=1.0 if all(c.paper_id and c.chunk_id for c in answer.citations) else 0.0,
        page_ok=page_ok,
        page_total=page_total,
        n_claims=n_claims,
        n_unsupported_claims=n_unsupported,
    )


def compute_citation_metrics_final(question: EvalQuestion, answer: FinalAnswer) -> CitationMetrics:
    papers = _papers_from_final(answer)
    page_ok, page_total = _page_traceable_final(answer)
    report = answer.citation_report
    validity = 1.0 if report is None else (1.0 if report.is_valid else 0.0)
    n_claims = max(len(answer.claims), 1 if answer.markdown.strip() else 0)
    # Claims without evidence_ids after validation should be zero; count draft issues
    n_unsupported = 0
    if report is not None:
        n_unsupported = sum(
            1 for i in report.issues if i.severity == "error" and "no valid citations" in i.message
        )
    return compute_citation_metrics_from_papers(
        question,
        papers,
        validity_rate=validity,
        page_ok=page_ok,
        page_total=page_total,
        n_claims=n_claims,
        n_unsupported_claims=n_unsupported,
    )


def aggregate_citation_metrics(
    rows: list[tuple[str, CitationMetrics]],
) -> CitationMetricAggregate:
    if not rows:
        return CitationMetricAggregate()
    keys = [
        "citation_precision",
        "citation_recall",
        "citation_validity_rate",
        "page_traceability_rate",
        "unsupported_claim_rate",
    ]
    n = len(rows)
    totals = {k: 0.0 for k in keys}
    by_type_sums: dict[str, dict[str, float]] = {}
    by_type_n: dict[str, int] = {}
    for qtype, m in rows:
        for k in keys:
            totals[k] += float(getattr(m, k))
        bucket = by_type_sums.setdefault(qtype, {k: 0.0 for k in keys})
        for k in keys:
            bucket[k] += float(getattr(m, k))
        by_type_n[qtype] = by_type_n.get(qtype, 0) + 1
    by_type = {
        qtype: {**{k: s[k] / max(1, by_type_n[qtype]) for k in keys}, "n": float(by_type_n[qtype])}
        for qtype, s in by_type_sums.items()
    }
    return CitationMetricAggregate(
        n=n,
        citation_precision=totals["citation_precision"] / n,
        citation_recall=totals["citation_recall"] / n,
        citation_validity_rate=totals["citation_validity_rate"] / n,
        page_traceability_rate=totals["page_traceability_rate"] / n,
        unsupported_claim_rate=totals["unsupported_claim_rate"] / n,
        by_type=by_type,
    )


def citations_to_naive_refs(citations: Iterable[CitationRef]) -> list[CitationRef]:
    return list(citations)

"""Deterministic retrieval metrics (Recall@K, MRR, nDCG@K)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from pydantic import BaseModel, Field

from scholar_agent.evaluation.dataset import EvalQuestion
from scholar_agent.models.retrieval import RetrievalHit


class RetrievalMetrics(BaseModel):
    recall_at_k: float = 0.0
    recall_at_k_paper: float = 0.0
    mrr: float = 0.0
    mrr_paper: float = 0.0
    ndcg_at_k: float = 0.0
    hit_at_k: float = 0.0
    k: int = 0
    n_gold_chunks: int = 0
    n_gold_papers: int = 0


class RetrievalMetricAggregate(BaseModel):
    n: int = 0
    recall_at_k: float = 0.0
    recall_at_k_paper: float = 0.0
    mrr: float = 0.0
    mrr_paper: float = 0.0
    ndcg_at_k: float = 0.0
    hit_at_k: float = 0.0
    by_type: dict[str, dict[str, float]] = Field(default_factory=dict)


def _first_rank(ids: list[str], gold: set[str]) -> int | None:
    for i, item in enumerate(ids, start=1):
        if item in gold:
            return i
    return None


def _dcg(relevances: list[float]) -> float:
    total = 0.0
    for i, rel in enumerate(relevances, start=1):
        if rel <= 0:
            continue
        total += (2**rel - 1) / math.log2(i + 1)
    return total


def graded_relevance_map(question: EvalQuestion) -> dict[str, float]:
    """Map chunk_id → graded relevance (default 1.0 for required ids)."""
    grades: dict[str, float] = {}
    for cid in question.required_chunk_ids:
        grades[cid] = max(grades.get(cid, 0.0), 1.0)
    for g in question.gold_evidence:
        if g.chunk_id:
            grades[g.chunk_id] = max(grades.get(g.chunk_id, 0.0), float(g.relevance))
    return grades


def compute_retrieval_metrics(
    question: EvalQuestion,
    hits: Iterable[RetrievalHit],
    *,
    k: int = 10,
) -> RetrievalMetrics:
    """Compute retrieval metrics against gold chunk/paper IDs.

    Unanswerable questions with no gold return perfect scores when hits are empty,
    otherwise zero (false retrieval for unanswerable).
    """
    hit_list = list(hits)[:k]
    gold_chunks = question.gold_chunk_ids()
    gold_papers = question.gold_paper_ids()

    if question.unanswerable and not gold_chunks and not gold_papers:
        # Prefer empty or irrelevant retrieval; do not reward retrieving anything
        empty = len(hit_list) == 0
        score = 1.0 if empty else 0.0
        return RetrievalMetrics(
            recall_at_k=score,
            recall_at_k_paper=score,
            mrr=score,
            mrr_paper=score,
            ndcg_at_k=score,
            hit_at_k=score,
            k=k,
            n_gold_chunks=0,
            n_gold_papers=0,
        )

    retrieved_chunks = [h.chunk_id for h in hit_list]
    retrieved_papers = [h.paper_id for h in hit_list]

    if gold_chunks:
        inter = gold_chunks.intersection(retrieved_chunks)
        recall_chunk = len(inter) / len(gold_chunks)
        rank_c = _first_rank(retrieved_chunks, gold_chunks)
        mrr_c = 1.0 / rank_c if rank_c else 0.0
        hit_c = 1.0 if rank_c else 0.0
    else:
        recall_chunk = 0.0
        mrr_c = 0.0
        hit_c = 0.0

    if gold_papers:
        inter_p = gold_papers.intersection(retrieved_papers)
        recall_p = len(inter_p) / len(gold_papers)
        rank_p = _first_rank(retrieved_papers, gold_papers)
        mrr_p = 1.0 / rank_p if rank_p else 0.0
        hit_p = 1.0 if rank_p else 0.0
    else:
        recall_p = 0.0
        mrr_p = 0.0
        hit_p = 0.0

    grades = graded_relevance_map(question)
    rels = [grades.get(h.chunk_id, 0.0) for h in hit_list]
    dcg = _dcg(rels)
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = _dcg(ideal) if ideal else 0.0
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return RetrievalMetrics(
        recall_at_k=recall_chunk if gold_chunks else recall_p,
        recall_at_k_paper=recall_p,
        mrr=mrr_c if gold_chunks else mrr_p,
        mrr_paper=mrr_p,
        ndcg_at_k=ndcg if gold_chunks else hit_p,
        hit_at_k=hit_c if gold_chunks else hit_p,
        k=k,
        n_gold_chunks=len(gold_chunks),
        n_gold_papers=len(gold_papers),
    )


def aggregate_retrieval_metrics(
    rows: list[tuple[str, RetrievalMetrics]],
) -> RetrievalMetricAggregate:
    if not rows:
        return RetrievalMetricAggregate()
    n = len(rows)
    keys = [
        "recall_at_k",
        "recall_at_k_paper",
        "mrr",
        "mrr_paper",
        "ndcg_at_k",
        "hit_at_k",
    ]
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
    by_type: dict[str, dict[str, float]] = {}
    for qtype, sums in by_type_sums.items():
        cnt = max(1, by_type_n[qtype])
        by_type[qtype] = {k: sums[k] / cnt for k in keys}
        by_type[qtype]["n"] = float(cnt)
    return RetrievalMetricAggregate(
        n=n,
        recall_at_k=totals["recall_at_k"] / n,
        recall_at_k_paper=totals["recall_at_k_paper"] / n,
        mrr=totals["mrr"] / n,
        mrr_paper=totals["mrr_paper"] / n,
        ndcg_at_k=totals["ndcg_at_k"] / n,
        hit_at_k=totals["hit_at_k"] / n,
        by_type=by_type,
    )

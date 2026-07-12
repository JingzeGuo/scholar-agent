"""Answer-side metrics: deterministic claim overlap + optional RAGAS."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from scholar_agent.evaluation.dataset import EvalQuestion
from scholar_agent.ids import normalize_text

_STOP = {
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
    "what",
    "how",
    "that",
    "this",
}


class AnswerMetrics(BaseModel):
    claim_overlap: float = 0.0
    token_f1: float = 0.0
    refusal_correct: float = 0.0
    faithfulness_proxy: float = 0.0
    ragas_faithfulness: float | None = None
    ragas_answer_relevancy: float | None = None
    used_ragas: bool = False


class AnswerMetricAggregate(BaseModel):
    n: int = 0
    claim_overlap: float = 0.0
    token_f1: float = 0.0
    refusal_correct: float = 0.0
    faithfulness_proxy: float = 0.0
    ragas_faithfulness: float | None = None
    ragas_answer_relevancy: float | None = None
    by_type: dict[str, dict[str, float]] = Field(default_factory=dict)


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", normalize_text(text))
        if t not in _STOP and len(t) > 1
    }


def token_f1(pred: str, ref: str) -> float:
    p = _tokens(pred)
    r = _tokens(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    inter = len(p & r)
    precision = inter / len(p)
    recall = inter / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def claim_overlap_score(pred: str, claims: list[str]) -> float:
    if not claims:
        return token_f1(pred, "") if not pred.strip() else 0.0
    scores = [token_f1(pred, c) for c in claims]
    return max(scores) if scores else 0.0


_REFUSAL_CUES = (
    "cannot answer",
    "can't answer",
    "corpus does not",
    "not contain",
    "insufficient",
    "no supporting",
    "limitation",
    "unanswerable",
    "no verified evidence",
    "does not provide",
    "not available",
)


def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(cue in low for cue in _REFUSAL_CUES)


def faithfulness_proxy(answer: str, contexts: list[str]) -> float:
    """Fraction of answer tokens covered by retrieved/evidence contexts."""
    ans_toks = _tokens(answer)
    if not ans_toks:
        return 1.0
    if not contexts:
        return 0.0
    ctx = _tokens(" ".join(contexts))
    if not ctx:
        return 0.0
    return len(ans_toks & ctx) / len(ans_toks)


def compute_answer_metrics(
    question: EvalQuestion,
    answer_text: str,
    *,
    contexts: list[str] | None = None,
    use_ragas: bool = False,
) -> AnswerMetrics:
    contexts = contexts or []
    if question.unanswerable:
        refusal = 1.0 if is_refusal(answer_text) else 0.0
        return AnswerMetrics(
            claim_overlap=refusal,
            token_f1=refusal,
            refusal_correct=refusal,
            faithfulness_proxy=1.0 if refusal else faithfulness_proxy(answer_text, contexts),
            used_ragas=False,
        )

    claims = question.reference_claims or (
        [question.reference_answer] if question.reference_answer else []
    )
    overlap = claim_overlap_score(answer_text, claims)
    f1 = token_f1(answer_text, question.reference_answer or " ".join(claims))
    faith = faithfulness_proxy(answer_text, contexts)
    metrics = AnswerMetrics(
        claim_overlap=overlap,
        token_f1=f1,
        refusal_correct=1.0,  # N/A for answerable; treat as neutral 1.0
        faithfulness_proxy=faith,
        used_ragas=False,
    )
    if use_ragas:
        ragas_scores = try_ragas_scores(
            question=question.question,
            answer=answer_text,
            contexts=contexts,
            reference=question.reference_answer,
        )
        if ragas_scores is not None:
            metrics.ragas_faithfulness = ragas_scores.get("faithfulness")
            metrics.ragas_answer_relevancy = ragas_scores.get("answer_relevancy")
            metrics.used_ragas = True
    return metrics


def try_ragas_scores(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
) -> dict[str, float] | None:
    """Optional RAGAS integration. Returns None if ragas is unavailable."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except Exception:
        return None

    try:
        ds = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts or [""]],
                "ground_truth": [reference or answer],
            }
        )
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
        # ragas returns different types across versions
        row: dict[str, Any]
        if hasattr(result, "to_pandas"):
            row = result.to_pandas().iloc[0].to_dict()
        elif isinstance(result, dict):
            row = dict(result)
        else:
            row = dict(result)
        out: dict[str, float] = {}
        for key in ("faithfulness", "answer_relevancy"):
            if key in row and row[key] is not None:
                try:
                    out[key] = float(row[key])
                except (TypeError, ValueError):
                    continue
        return out or None
    except Exception:
        return None


def aggregate_answer_metrics(
    rows: list[tuple[str, AnswerMetrics]],
) -> AnswerMetricAggregate:
    if not rows:
        return AnswerMetricAggregate()
    keys = ["claim_overlap", "token_f1", "refusal_correct", "faithfulness_proxy"]
    n = len(rows)
    totals = {k: 0.0 for k in keys}
    ragas_f: list[float] = []
    ragas_r: list[float] = []
    by_type_sums: dict[str, dict[str, float]] = {}
    by_type_n: dict[str, int] = {}
    for qtype, m in rows:
        for k in keys:
            totals[k] += float(getattr(m, k))
        if m.ragas_faithfulness is not None:
            ragas_f.append(m.ragas_faithfulness)
        if m.ragas_answer_relevancy is not None:
            ragas_r.append(m.ragas_answer_relevancy)
        bucket = by_type_sums.setdefault(qtype, {k: 0.0 for k in keys})
        for k in keys:
            bucket[k] += float(getattr(m, k))
        by_type_n[qtype] = by_type_n.get(qtype, 0) + 1
    by_type = {
        qtype: {
            **{k: s[k] / max(1, by_type_n[qtype]) for k in keys},
            "n": float(by_type_n[qtype]),
        }
        for qtype, s in by_type_sums.items()
    }
    return AnswerMetricAggregate(
        n=n,
        claim_overlap=totals["claim_overlap"] / n,
        token_f1=totals["token_f1"] / n,
        refusal_correct=totals["refusal_correct"] / n,
        faithfulness_proxy=totals["faithfulness_proxy"] / n,
        ragas_faithfulness=(sum(ragas_f) / len(ragas_f)) if ragas_f else None,
        ragas_answer_relevancy=(sum(ragas_r) / len(ragas_r)) if ragas_r else None,
        by_type=by_type,
    )


def ragas_available() -> bool:
    try:
        import datasets  # noqa: F401
        import ragas  # noqa: F401

        return True
    except Exception:
        return False


def describe_ragas_status() -> dict[str, Any]:
    return {"available": ragas_available()}

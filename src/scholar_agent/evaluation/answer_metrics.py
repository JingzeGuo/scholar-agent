"""Answer-side metrics: deterministic claim overlap + optional RAGAS."""

from __future__ import annotations

import re
from collections.abc import Callable
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from scholar_agent.evaluation.dataset import EvalQuestion
from scholar_agent.ids import normalize_text

RagasMetricName = Literal["faithfulness", "answer_relevancy"]


class RagasMetricFailure(BaseModel):
    """Secret-free diagnostic for one optional RAGAS metric failure."""

    metric: RagasMetricName
    code: str
    exception_type: str | None = None


class RagasEvaluationResult(BaseModel):
    """Structured boundary for one RAGAS evaluation attempt.

    Partial success is intentional: a parser or provider failure in one metric
    must not discard a valid score from the other metric.
    """

    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)
    failures: list[RagasMetricFailure] = Field(default_factory=list)
    cached: bool = False

    @model_validator(mode="after")
    def _require_score_or_failure(self) -> RagasEvaluationResult:
        if self.faithfulness is None and self.answer_relevancy is None and not self.failures:
            raise ValueError("a RAGAS result requires a score or structured failure")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> Literal["success", "partial", "failed"]:
        scores = self.as_scores()
        if scores and self.failures:
            return "partial"
        if scores:
            return "success"
        return "failed"

    def as_scores(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        if self.faithfulness is not None:
            scores["faithfulness"] = self.faithfulness
        if self.answer_relevancy is not None:
            scores["answer_relevancy"] = self.answer_relevancy
        return scores


RagasEvaluator = Callable[..., RagasEvaluationResult | dict[str, float] | None]

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
    claim_correctness: float = 0.0
    completeness: float = 0.0
    token_f1: float = 0.0
    refusal_correct: float = 0.0
    faithfulness_proxy: float = 0.0
    ragas_faithfulness: float | None = None
    ragas_answer_relevancy: float | None = None
    ragas_status: Literal["success", "partial", "failed"] | None = None
    ragas_cached: bool = False
    ragas_failures: list[RagasMetricFailure] = Field(default_factory=list)
    contradiction_handling_accuracy: float | None = None
    used_ragas: bool = False


class AnswerMetricAggregate(BaseModel):
    n: int = 0
    claim_overlap: float = 0.0
    claim_correctness: float = 0.0
    completeness: float = 0.0
    token_f1: float = 0.0
    refusal_correct: float = 0.0
    faithfulness_proxy: float = 0.0
    ragas_faithfulness: float | None = None
    ragas_answer_relevancy: float | None = None
    contradiction_handling_accuracy: float | None = None
    contradiction_coverage_rate: float = 0.0
    by_type: dict[str, dict[str, float]] = Field(default_factory=dict)


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", normalize_text(text)) if t not in _STOP and len(t) > 1
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


def claim_correctness_score(pred: str, claims: list[str]) -> float:
    """Lexical claim-level precision proxy, averaged over predicted units.

    This deterministic score is intentionally named and documented as a proxy:
    it measures whether each answer claim has a close reference-claim match; it
    is not a substitute for expert semantic fact checking.
    """
    if not claims:
        return 1.0 if not pred.strip() else 0.0
    units = _claim_units(pred)
    if not units:
        return 0.0
    return sum(max(token_f1(unit, claim) for claim in claims) for unit in units) / len(units)


def completeness_score(pred: str, claims: list[str]) -> float:
    """Average lexical recall of the manually supplied reference claims."""
    if not claims:
        return 1.0 if not pred.strip() else 0.0
    pred_tokens = _tokens(" ".join(_claim_units(pred)))
    scores: list[float] = []
    for claim in claims:
        reference = _tokens(claim)
        if not reference:
            scores.append(1.0)
        elif not pred_tokens:
            scores.append(0.0)
        else:
            scores.append(len(pred_tokens & reference) / len(reference))
    return sum(scores) / len(scores)


def contradiction_handling_score(
    answer: str,
    *,
    contradiction_expected: bool,
    contradiction_detected: bool,
) -> float | None:
    """Score explicit detection + surfacing only when a conflict is observable.

    ``None`` means the row has no contradiction annotation or detected conflict,
    so reports can expose metric coverage instead of fabricating a perfect zero
    or one for non-applicable questions.
    """
    if not contradiction_expected and not contradiction_detected:
        return None
    surfaced = bool(
        re.search(
            r"\b(contradict(?:ion|ory)?|conflict(?:ing)?|disagree(?:ment)?|"
            r"inconsisten(?:t|cy)|sources? differ|mixed evidence)\b",
            answer,
            flags=re.I,
        )
    )
    detection_ok = contradiction_detected if contradiction_expected else True
    return 1.0 if detection_ok and surfaced else 0.0


def _strip_citations(text: str) -> str:
    return re.sub(r"\[[^\]\n]+\]", " ", text)


def _claim_units(text: str) -> list[str]:
    cleaned = _strip_citations(text)
    units: list[str] = []
    for line in cleaned.splitlines():
        line = re.sub(r"^\s*[-*#]+\s*", "", line).strip()
        lowered = line.lower()
        if lowered in {"answer", "evidence", "references", "sources", "source cards"}:
            continue
        if not line or lowered.startswith(
            (
                "question:",
                "sources:",
                "evidence-based notes",
                "evidence notes",
                "these passages are ranked",
            )
        ):
            continue
        for part in re.split(r"(?<=[.!?])\s+|\s*;\s*", line):
            part = part.strip()
            if len(_tokens(part)) >= 2:
                units.append(part)
    return units


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
    ragas_evaluator: RagasEvaluator | None = None,
) -> AnswerMetrics:
    contexts = contexts or []
    if question.unanswerable:
        refusal = 1.0 if is_refusal(answer_text) else 0.0
        return AnswerMetrics(
            claim_overlap=refusal,
            claim_correctness=refusal,
            completeness=refusal,
            token_f1=refusal,
            refusal_correct=refusal,
            faithfulness_proxy=1.0 if refusal else faithfulness_proxy(answer_text, contexts),
            used_ragas=False,
        )

    claims = question.reference_claims or (
        [question.reference_answer] if question.reference_answer else []
    )
    overlap = claim_overlap_score(answer_text, claims)
    correctness = claim_correctness_score(answer_text, claims)
    completeness = completeness_score(answer_text, claims)
    f1 = token_f1(answer_text, question.reference_answer or " ".join(claims))
    faith = faithfulness_proxy(answer_text, contexts)
    metrics = AnswerMetrics(
        claim_overlap=overlap,
        claim_correctness=correctness,
        completeness=completeness,
        token_f1=f1,
        refusal_correct=1.0,  # N/A for answerable; treat as neutral 1.0
        faithfulness_proxy=faith,
        used_ragas=False,
    )
    if use_ragas and ragas_evaluator is not None:
        try:
            raw_result = ragas_evaluator(
                question=question.question,
                answer=answer_text,
                contexts=contexts,
                reference=question.reference_answer,
            )
        except Exception as exc:  # noqa: BLE001
            # Provider exceptions can contain response bodies. Persist only the
            # exception class, never the message or raw response.
            outcome = RagasEvaluationResult(
                failures=[
                    RagasMetricFailure(
                        metric="faithfulness",
                        code="evaluator_exception",
                        exception_type=type(exc).__name__,
                    ),
                    RagasMetricFailure(
                        metric="answer_relevancy",
                        code="evaluator_exception",
                        exception_type=type(exc).__name__,
                    ),
                ]
            )
        else:
            outcome = _coerce_ragas_result(raw_result)
        if outcome is not None:
            metrics.ragas_faithfulness = outcome.faithfulness
            metrics.ragas_answer_relevancy = outcome.answer_relevancy
            metrics.ragas_status = outcome.status
            metrics.ragas_cached = outcome.cached
            metrics.ragas_failures = list(outcome.failures)
            metrics.used_ragas = bool(outcome.as_scores())
    return metrics


def _coerce_ragas_result(
    result: RagasEvaluationResult | dict[str, float] | None,
) -> RagasEvaluationResult:
    """Accept the structured runtime result and legacy explicit test delegates."""
    if result is None:
        return RagasEvaluationResult(
            failures=[
                RagasMetricFailure(metric="faithfulness", code="no_result"),
                RagasMetricFailure(metric="answer_relevancy", code="no_result"),
            ]
        )
    if isinstance(result, RagasEvaluationResult):
        return result
    scores: dict[str, float] = {}
    failures: list[RagasMetricFailure] = []
    for metric in ("faithfulness", "answer_relevancy"):
        value = result.get(metric)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            failures.append(RagasMetricFailure(metric=metric, code="invalid_score_type"))
            continue
        if not isfinite(numeric):
            failures.append(RagasMetricFailure(metric=metric, code="non_finite_score"))
            continue
        if not 0.0 <= numeric <= 1.0:
            failures.append(RagasMetricFailure(metric=metric, code="out_of_range_score"))
            continue
        scores[metric] = numeric
    if not scores and not failures:
        failures = [
            RagasMetricFailure(metric="faithfulness", code="missing_score"),
            RagasMetricFailure(metric="answer_relevancy", code="missing_score"),
        ]
    return RagasEvaluationResult(
        faithfulness=scores.get("faithfulness"),
        answer_relevancy=scores.get("answer_relevancy"),
        failures=failures,
    )


def try_ragas_scores(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
    llm: Any,
    embeddings: Any,
) -> RagasEvaluationResult:
    """Score one answer with explicitly configured RAGAS models.

    ``llm`` and ``embeddings`` are mandatory by design. RAGAS otherwise falls
    back to its OpenAI defaults, which could use the wrong provider and make an
    unplanned paid call.
    """
    scores: dict[str, float] = {}
    failures: list[RagasMetricFailure] = []
    for metric in ("faithfulness", "answer_relevancy"):
        try:
            raw_score = _run_ragas_metric(
                metric,
                question=question,
                answer=answer,
                contexts=contexts,
                reference=reference,
                llm=llm,
                embeddings=embeddings,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                RagasMetricFailure(
                    metric=metric,
                    code=_ragas_exception_code(exc),
                    exception_type=type(exc).__name__,
                )
            )
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            failures.append(RagasMetricFailure(metric=metric, code="invalid_score_type"))
            continue
        if not isfinite(score):
            failures.append(RagasMetricFailure(metric=metric, code="non_finite_score"))
            continue
        if not 0.0 <= score <= 1.0:
            failures.append(RagasMetricFailure(metric=metric, code="out_of_range_score"))
            continue
        scores[metric] = score
    return RagasEvaluationResult(
        faithfulness=scores.get("faithfulness"),
        answer_relevancy=scores.get("answer_relevancy"),
        failures=failures,
    )


def _run_ragas_metric(
    metric: RagasMetricName,
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
    llm: Any,
    embeddings: Any,
) -> float:
    """Run one RAGAS 0.3 metric in isolation with explicit LangChain adapters."""
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, Faithfulness

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts or [""],
        reference=reference or answer,
    )
    if metric == "faithfulness":
        scorer = Faithfulness(llm=ragas_llm)
    else:
        scorer = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    return float(scorer.single_turn_score(sample))


def _ragas_exception_code(exc: Exception) -> str:
    """Map provider/library exceptions to stable codes without saving messages."""
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "authentication" in name or "permission" in name:
        return "provider_authentication"
    if "ratelimit" in name or "rate_limit" in name:
        return "provider_rate_limit"
    if "parser" in name or "parse" in name:
        return "output_parse_failed"
    if "connection" in name:
        return "provider_connection"
    return "metric_exception"


def aggregate_answer_metrics(
    rows: list[tuple[str, AnswerMetrics]],
) -> AnswerMetricAggregate:
    if not rows:
        return AnswerMetricAggregate()
    keys = [
        "claim_overlap",
        "claim_correctness",
        "completeness",
        "token_f1",
        "refusal_correct",
        "faithfulness_proxy",
    ]
    n = len(rows)
    totals = {k: 0.0 for k in keys}
    ragas_f: list[float] = []
    ragas_r: list[float] = []
    contradiction: list[float] = []
    by_type_sums: dict[str, dict[str, float]] = {}
    by_type_n: dict[str, int] = {}
    for qtype, m in rows:
        for k in keys:
            totals[k] += float(getattr(m, k))
        if m.ragas_faithfulness is not None:
            ragas_f.append(m.ragas_faithfulness)
        if m.ragas_answer_relevancy is not None:
            ragas_r.append(m.ragas_answer_relevancy)
        if m.contradiction_handling_accuracy is not None:
            contradiction.append(m.contradiction_handling_accuracy)
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
        claim_correctness=totals["claim_correctness"] / n,
        completeness=totals["completeness"] / n,
        token_f1=totals["token_f1"] / n,
        refusal_correct=totals["refusal_correct"] / n,
        faithfulness_proxy=totals["faithfulness_proxy"] / n,
        ragas_faithfulness=(sum(ragas_f) / len(ragas_f)) if ragas_f else None,
        ragas_answer_relevancy=(sum(ragas_r) / len(ragas_r)) if ragas_r else None,
        contradiction_handling_accuracy=(
            (sum(contradiction) / len(contradiction)) if contradiction else None
        ),
        contradiction_coverage_rate=len(contradiction) / n if n else 0.0,
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

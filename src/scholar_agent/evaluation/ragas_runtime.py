"""Explicit provider wiring for the optional RAGAS metrics."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any, Self

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scholar_agent.config import LLMConfig
from scholar_agent.evaluation.answer_metrics import (
    RagasEvaluationResult,
    RagasEvaluator,
    RagasMetricFailure,
    try_ragas_scores,
)
from scholar_agent.retrieval.embeddings import Embedder
from scholar_agent.storage.cache import DiskCache

RAGAS_CACHE_SCHEMA = "ragas-metrics-v2"
RAGAS_CACHE_NAMESPACE = "ragas_metrics"


class RagasCacheIdentity(BaseModel):
    """Secret-free provider/model identity included in metric cache keys."""

    provider: str
    base_url: str
    model: str
    embedding_model: str


class RagasEvaluationInput(BaseModel):
    """Validated boundary for one paid evaluator call."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    contexts: list[str]
    reference: str


class CachedRagasScores(BaseModel):
    """Allowlisted numeric cache value; raw provider payloads cannot enter it."""

    model_config = ConfigDict(extra="ignore")

    value_schema: str = RAGAS_CACHE_SCHEMA
    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _at_least_one_metric(self) -> Self:
        if self.faithfulness is None and self.answer_relevancy is None:
            raise ValueError("at least one supported RAGAS metric is required")
        if self.value_schema != RAGAS_CACHE_SCHEMA:
            raise ValueError("RAGAS cache value schema mismatch")
        return self

    def as_evaluator_result(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        if self.faithfulness is not None:
            scores["faithfulness"] = self.faithfulness
        if self.answer_relevancy is not None:
            scores["answer_relevancy"] = self.answer_relevancy
        return scores

    def as_structured_result(self) -> RagasEvaluationResult:
        failures: list[RagasMetricFailure] = []
        if self.faithfulness is None:
            failures.append(
                RagasMetricFailure(metric="faithfulness", code="metric_unavailable_in_cache")
            )
        if self.answer_relevancy is None:
            failures.append(
                RagasMetricFailure(metric="answer_relevancy", code="metric_unavailable_in_cache")
            )
        return RagasEvaluationResult(
            faithfulness=self.faithfulness,
            answer_relevancy=self.answer_relevancy,
            failures=failures,
            cached=True,
        )


class ScholarEmbeddings(Embeddings):  # type: ignore[misc]
    """Expose ScholarAgent's selected embedder through LangChain's interface."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)


def create_ragas_evaluator(
    llm_config: LLMConfig,
    embedder: Embedder,
    *,
    cache_dir: Path | None = None,
) -> tuple[RagasEvaluator | None, dict[str, Any]]:
    """Build a RAGAS scorer without falling back to implicit provider defaults."""
    try:
        import datasets  # noqa: F401
        import ragas  # noqa: F401
    except Exception as exc:
        return None, {
            "available": False,
            "configured": False,
            "reason": f"optional dependencies unavailable: {type(exc).__name__}",
        }

    if not llm_config.api_key:
        return None, {
            "available": True,
            "configured": False,
            "reason": "DEEPSEEK_API_KEY/OPENAI_API_KEY is not configured",
        }

    llm = ChatOpenAI(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        model=llm_config.fast_model,
        temperature=0.0,
        max_retries=llm_config.max_retries,
        timeout=llm_config.request_timeout_s,
    )
    embeddings = ScholarEmbeddings(embedder)

    def score_uncached(**kwargs: Any) -> RagasEvaluationResult:
        return try_ragas_scores(llm=llm, embeddings=embeddings, **kwargs)

    status: dict[str, Any] = {
        "available": True,
        "configured": True,
        "provider": llm_config.provider,
        "base_url": llm_config.base_url,
        "model": llm_config.fast_model,
        "embedding_model": embedder.model_name,
        "cache_schema": RAGAS_CACHE_SCHEMA,
        "evaluation_status": "not_run",
        "evaluation_attempts": 0,
        "cached_evaluations": 0,
        "successful_metric_counts": {},
        "metric_failure_counts": {},
        "metric_failure_details": [],
        "reason": None,
    }
    cache = DiskCache(
        root=cache_dir or Path("data/evaluation/.cache"),
        namespace=RAGAS_CACHE_NAMESPACE,
        schema_version=RAGAS_CACHE_SCHEMA,
    )
    status["cache_dir"] = str(cache.root / cache.namespace)
    evaluator = create_cached_ragas_evaluator(
        score_uncached,
        cache=cache,
        identity=RagasCacheIdentity(
            provider=llm_config.provider,
            base_url=llm_config.base_url,
            model=llm_config.fast_model,
            embedding_model=embedder.model_name,
        ),
        status=status,
    )
    return evaluator, status


def create_cached_ragas_evaluator(
    evaluator: RagasEvaluator,
    *,
    cache: DiskCache,
    identity: RagasCacheIdentity,
    status: dict[str, Any] | None = None,
) -> RagasEvaluator:
    """Cache allowlisted RAGAS numbers, never raw responses or credentials.

    Inputs are present only in the SHA-256 cache-key material; ``DiskCache``
    persists the digest and the validated numeric value, not that material.
    """

    def update_status() -> None:
        if status is not None:
            status.update(
                {
                    "cache_hits": cache.stats.hits,
                    "cache_misses": cache.stats.misses,
                    "cache_stores": cache.stats.stores,
                    "cache_invalidations": cache.stats.invalidations,
                    "cache_corruptions": cache.stats.corruptions,
                }
            )

    update_status()

    def publish_result(result: RagasEvaluationResult) -> None:
        if status is None:
            return
        status["evaluation_attempts"] = int(status.get("evaluation_attempts", 0)) + 1
        if result.cached:
            status["cached_evaluations"] = int(status.get("cached_evaluations", 0)) + 1
        successes = dict(status.get("successful_metric_counts") or {})
        for metric in result.as_scores():
            successes[metric] = int(successes.get(metric, 0)) + 1
        status["successful_metric_counts"] = successes
        failure_counts = dict(status.get("metric_failure_counts") or {})
        details = list(status.get("metric_failure_details") or [])
        for failure in result.failures:
            key = f"{failure.metric}:{failure.code}"
            failure_counts[key] = int(failure_counts.get(key, 0)) + 1
            detail = failure.model_dump(mode="json")
            if detail not in details:
                details.append(detail)
        status["metric_failure_counts"] = failure_counts
        status["metric_failure_details"] = details
        status["last_evaluation_status"] = result.status
        if failure_counts:
            status["evaluation_status"] = "partial" if any(successes.values()) else "failed"
            status["reason"] = "one or more RAGAS metrics failed; see metric_failure_details"
        else:
            status["evaluation_status"] = "success"
            status["reason"] = None

    def score(**kwargs: Any) -> RagasEvaluationResult:
        request = RagasEvaluationInput.model_validate(kwargs)
        key = cache.make_key(
            {
                "identity": identity.model_dump(mode="json"),
                "request": request.model_dump(mode="json"),
                "metrics": ["faithfulness", "answer_relevancy"],
            }
        )
        cached = cache.get(key)
        if cached is not None:
            try:
                validated_cached = CachedRagasScores.model_validate(cached)
            except ValueError:
                cache.delete(key)
            else:
                update_status()
                result = validated_cached.as_structured_result()
                publish_result(result)
                return result

        try:
            raw_result = evaluator(**request.model_dump())
        except Exception as exc:  # noqa: BLE001
            # Persist the class only: provider exception messages can contain
            # request/response data or credentials.
            result = _failed_result("evaluator_exception", type(exc).__name__)
        else:
            result = _normalize_evaluator_result(raw_result)
        scores = result.as_scores()
        if not scores:
            update_status()
            publish_result(result)
            return result
        validated = CachedRagasScores.model_validate(scores)
        cache.set(key, validated.model_dump(mode="json", exclude_none=True))
        update_status()
        publish_result(result)
        return result

    return score


def _normalize_evaluator_result(
    value: RagasEvaluationResult | Mapping[str, Any] | None,
) -> RagasEvaluationResult:
    """Validate each allowlisted metric independently so partial results survive."""
    if isinstance(value, RagasEvaluationResult):
        return value
    if value is None:
        return _failed_result("no_result")
    scores: dict[str, float] = {}
    failures: list[RagasMetricFailure] = []
    for metric in ("faithfulness", "answer_relevancy"):
        raw = value.get(metric)
        if raw is None:
            failures.append(RagasMetricFailure(metric=metric, code="missing_score"))
            continue
        try:
            numeric = float(raw)
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
    return RagasEvaluationResult(
        faithfulness=scores.get("faithfulness"),
        answer_relevancy=scores.get("answer_relevancy"),
        failures=failures,
    )


def _failed_result(code: str, exception_type: str | None = None) -> RagasEvaluationResult:
    return RagasEvaluationResult(
        failures=[
            RagasMetricFailure(metric="faithfulness", code=code, exception_type=exception_type),
            RagasMetricFailure(metric="answer_relevancy", code=code, exception_type=exception_type),
        ]
    )

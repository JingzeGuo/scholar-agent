"""Secret-free, versioned caching for optional paid RAGAS calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scholar_agent.evaluation import answer_metrics
from scholar_agent.evaluation.answer_metrics import RagasEvaluationResult
from scholar_agent.evaluation.ragas_runtime import (
    RAGAS_CACHE_NAMESPACE,
    RAGAS_CACHE_SCHEMA,
    RagasCacheIdentity,
    create_cached_ragas_evaluator,
)
from scholar_agent.storage.cache import DiskCache


def _identity(*, model: str = "deepseek-test") -> RagasCacheIdentity:
    return RagasCacheIdentity(
        provider="deepseek",
        base_url="https://api.deepseek.example",
        model=model,
        embedding_model="BAAI/bge-small-en-v1.5",
    )


def _request() -> dict[str, object]:
    return {
        "question": "What is Self-RAG?",
        "answer": "Self-RAG retrieves on demand.",
        "contexts": ["Self-RAG uses reflection tokens to retrieve passages on demand."],
        "reference": "Self-RAG adaptively retrieves evidence.",
    }


def test_cached_ragas_evaluator_calls_paid_delegate_once_and_updates_status(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def paid_delegate(**kwargs: Any) -> dict[str, float]:
        calls.append(kwargs)
        return {"faithfulness": 0.75, "answer_relevancy": 0.8}

    cache = DiskCache(
        root=tmp_path,
        namespace=RAGAS_CACHE_NAMESPACE,
        schema_version=RAGAS_CACHE_SCHEMA,
    )
    status: dict[str, Any] = {}
    evaluator = create_cached_ragas_evaluator(
        paid_delegate,
        cache=cache,
        identity=_identity(),
        status=status,
    )

    first = evaluator(**_request())
    second = evaluator(**_request())

    assert (
        first.as_scores()
        == second.as_scores()
        == {
            "faithfulness": 0.75,
            "answer_relevancy": 0.8,
        }
    )
    assert first.status == second.status == "success"
    assert first.cached is False
    assert second.cached is True
    assert len(calls) == 1
    assert status["cache_hits"] == 1
    assert status["cache_misses"] == 1
    assert status["cache_stores"] == 1


def test_ragas_cache_persists_only_allowlisted_numbers_not_inputs_or_raw_payload(
    tmp_path: Path,
) -> None:
    provider_secret = "opaque-paid-provider-secret"

    def paid_delegate(**_kwargs: Any) -> dict[str, Any]:
        return {
            "faithfulness": 0.9,
            "answer_relevancy": 0.7,
            "raw_provider_response": {
                "api_key": provider_secret,
                "reasoning": "sensitive provider output",
            },
        }

    cache = DiskCache(
        root=tmp_path,
        namespace=RAGAS_CACHE_NAMESPACE,
        schema_version=RAGAS_CACHE_SCHEMA,
    )
    evaluator = create_cached_ragas_evaluator(
        paid_delegate,  # type: ignore[arg-type]
        cache=cache,
        identity=_identity(),
    )
    result = evaluator(**_request())
    assert result.as_scores() == {"faithfulness": 0.9, "answer_relevancy": 0.7}

    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1
    raw = files[0].read_text(encoding="utf-8")
    record = json.loads(raw)
    assert provider_secret not in raw
    assert "raw_provider_response" not in raw
    assert "sensitive provider output" not in raw
    assert "What is Self-RAG?" not in raw
    assert "Self-RAG retrieves on demand" not in raw
    assert record["value"] == {
        "answer_relevancy": 0.7,
        "faithfulness": 0.9,
        "value_schema": RAGAS_CACHE_SCHEMA,
    }


def test_ragas_cache_key_includes_model_identity_and_schema(tmp_path: Path) -> None:
    calls = 0

    def paid_delegate(**_kwargs: Any) -> dict[str, float]:
        nonlocal calls
        calls += 1
        return {"faithfulness": 0.6}

    cache_v1 = DiskCache(
        root=tmp_path,
        namespace=RAGAS_CACHE_NAMESPACE,
        schema_version=RAGAS_CACHE_SCHEMA,
    )
    evaluator_a = create_cached_ragas_evaluator(
        paid_delegate,
        cache=cache_v1,
        identity=_identity(model="model-a"),
    )
    evaluator_b = create_cached_ragas_evaluator(
        paid_delegate,
        cache=cache_v1,
        identity=_identity(model="model-b"),
    )
    assert evaluator_a(**_request()).as_scores() == {"faithfulness": 0.6}
    assert evaluator_b(**_request()).as_scores() == {"faithfulness": 0.6}

    cache_v2 = DiskCache(
        root=tmp_path,
        namespace=RAGAS_CACHE_NAMESPACE,
        schema_version="ragas-metrics-v2-test",
    )
    evaluator_v2 = create_cached_ragas_evaluator(
        paid_delegate,
        cache=cache_v2,
        identity=_identity(model="model-a"),
    )
    assert evaluator_v2(**_request()).as_scores() == {"faithfulness": 0.6}
    assert calls == 3


def test_invalid_ragas_numbers_are_not_cached(tmp_path: Path) -> None:
    calls = 0

    def invalid_delegate(**_kwargs: Any) -> dict[str, float]:
        nonlocal calls
        calls += 1
        return {"faithfulness": float("nan")}

    cache = DiskCache(
        root=tmp_path,
        namespace=RAGAS_CACHE_NAMESPACE,
        schema_version=RAGAS_CACHE_SCHEMA,
    )
    evaluator = create_cached_ragas_evaluator(
        invalid_delegate,
        cache=cache,
        identity=_identity(),
    )
    first = evaluator(**_request())
    second = evaluator(**_request())
    assert first.status == second.status == "failed"
    assert {failure.code for failure in first.failures} == {
        "non_finite_score",
        "missing_score",
    }
    assert calls == 2
    assert cache.stats.stores == 0
    assert list(tmp_path.rglob("*.json")) == []


def test_partial_metric_success_is_returned_cached_and_reported(tmp_path: Path) -> None:
    calls = 0

    def partial_delegate(**_kwargs: Any) -> dict[str, float]:
        nonlocal calls
        calls += 1
        return {"faithfulness": 0.625, "answer_relevancy": float("nan")}

    cache = DiskCache(
        root=tmp_path,
        namespace=RAGAS_CACHE_NAMESPACE,
        schema_version=RAGAS_CACHE_SCHEMA,
    )
    status: dict[str, Any] = {}
    evaluator = create_cached_ragas_evaluator(
        partial_delegate,
        cache=cache,
        identity=_identity(),
        status=status,
    )

    first = evaluator(**_request())
    second = evaluator(**_request())

    assert first.as_scores() == second.as_scores() == {"faithfulness": 0.625}
    assert first.status == second.status == "partial"
    assert first.cached is False
    assert second.cached is True
    assert calls == 1
    assert cache.stats.stores == 1
    assert cache.stats.hits == 1
    assert status["evaluation_status"] == "partial"
    assert status["reason"] == ("one or more RAGAS metrics failed; see metric_failure_details")
    assert status["successful_metric_counts"] == {"faithfulness": 2}
    assert status["metric_failure_counts"] == {
        "answer_relevancy:non_finite_score": 1,
        "answer_relevancy:metric_unavailable_in_cache": 1,
    }


def test_ragas_metrics_are_isolated_and_exceptions_are_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-provider-secret-must-not-appear"

    def fake_metric(metric: str, **_kwargs: Any) -> float:
        if metric == "faithfulness":
            return 0.875
        raise RuntimeError(f"provider response contained {secret}")

    monkeypatch.setattr(answer_metrics, "_run_ragas_metric", fake_metric)
    result = answer_metrics.try_ragas_scores(
        question="question",
        answer="answer",
        contexts=["context"],
        reference="reference",
        llm=object(),
        embeddings=object(),
    )

    assert result.as_scores() == {"faithfulness": 0.875}
    assert result.status == "partial"
    assert result.failures == [
        answer_metrics.RagasMetricFailure(
            metric="answer_relevancy",
            code="metric_exception",
            exception_type="RuntimeError",
        )
    ]
    assert secret not in result.model_dump_json()


def test_cached_evaluator_converts_delegate_exception_to_structured_failure(
    tmp_path: Path,
) -> None:
    secret = "secret-response-body"

    def failing_delegate(**_kwargs: Any) -> RagasEvaluationResult:
        raise ValueError(secret)

    cache = DiskCache(
        root=tmp_path,
        namespace=RAGAS_CACHE_NAMESPACE,
        schema_version=RAGAS_CACHE_SCHEMA,
    )
    status: dict[str, Any] = {}
    evaluator = create_cached_ragas_evaluator(
        failing_delegate,
        cache=cache,
        identity=_identity(),
        status=status,
    )
    result = evaluator(**_request())

    assert result.status == "failed"
    assert all(failure.exception_type == "ValueError" for failure in result.failures)
    assert secret not in result.model_dump_json()
    assert secret not in json.dumps(status)
    assert list(tmp_path.rglob("*.json")) == []

"""One required cross-encoder rerank function."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from scholar_agent.indexes import ModelUnavailableError, resolve_model_path

ScoreFunction = Callable[[list[tuple[str, str]]], list[float]]


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str) -> Any:
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(resolve_model_path(model_name), local_files_only=True)
    except ModelUnavailableError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ModelUnavailableError(f"Reranker model failed: {model_name}") from exc


def rerank(
    queries: list[str],
    candidates: list[dict],
    model_name: str,
    *,
    scorer: ScoreFunction | None = None,
) -> list[dict]:
    """Score each query/chunk pair and keep each chunk's best query score."""
    queries = [query.strip() for query in queries if query.strip()]
    if not candidates or not queries:
        return []
    pairs = [(query, candidate["text"]) for candidate in candidates for query in queries]
    if scorer is not None:
        raw_scores = scorer(pairs)
    else:
        try:
            predicted: Any = _cross_encoder(model_name).predict(
                pairs,
                show_progress_bar=False,
            )
        except ModelUnavailableError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ModelUnavailableError(f"Reranker inference failed: {model_name}") from exc
        raw_scores = [float(score) for score in predicted]

    if len(raw_scores) != len(pairs):
        raise ValueError("Reranker returned an unexpected number of scores")
    width = len(queries)
    scores = [max(raw_scores[start : start + width]) for start in range(0, len(raw_scores), width)]
    scored = [
        {**item, "score": float(score)} for item, score in zip(candidates, scores, strict=True)
    ]
    return sorted(scored, key=lambda item: item["score"], reverse=True)

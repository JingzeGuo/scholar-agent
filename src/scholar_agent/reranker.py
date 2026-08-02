"""One cross-encoder rerank function with an offline lexical fallback."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from scholar_agent.indexes import resolve_model_path, tokenize

LOGGER = logging.getLogger(__name__)
ScoreFunction = Callable[[list[tuple[str, str]]], list[float]]


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(resolve_model_path(model_name), local_files_only=True)


def _lexical_scores(queries: list[str], candidates: list[dict]) -> list[float]:
    query_terms = [set(tokenize(query)) for query in queries]
    scores: list[float] = []
    for candidate in candidates:
        terms = set(tokenize(candidate["text"]))
        overlaps = [len(query.intersection(terms)) / len(query) for query in query_terms if query]
        best = max(overlaps, default=0.0)
        scores.append(best if best > 0 else float("-inf"))
    return scores


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
            raw_scores = [float(score) for score in predicted]
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            LOGGER.warning("[reranker] local cross-encoder unavailable; lexical fallback: %s", exc)
            raw_scores = []

    if raw_scores:
        width = len(queries)
        scores = [
            max(raw_scores[start : start + width]) for start in range(0, len(raw_scores), width)
        ]
    else:
        scores = _lexical_scores(queries, candidates)
    scored = [
        {**item, "score": float(score)} for item, score in zip(candidates, scores, strict=True)
    ]
    return sorted(scored, key=lambda item: item["score"], reverse=True)

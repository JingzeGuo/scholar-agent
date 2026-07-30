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


def _lexical_scores(question: str, candidates: list[dict]) -> list[float]:
    query_terms = set(tokenize(question))
    if not query_terms:
        return [0.0] * len(candidates)
    return [
        len(query_terms.intersection(tokenize(candidate["text"]))) / len(query_terms)
        for candidate in candidates
    ]


def rerank(
    question: str,
    candidates: list[dict],
    model_name: str,
    *,
    scorer: ScoreFunction | None = None,
) -> list[dict]:
    """Score question/chunk pairs and return the candidates in descending order."""
    if not candidates:
        return []
    pairs = [(question, candidate["text"]) for candidate in candidates]
    if scorer is not None:
        scores = scorer(pairs)
    else:
        try:
            raw_scores: Any = _cross_encoder(model_name).predict(
                pairs,
                show_progress_bar=False,
            )
            scores = [float(score) for score in raw_scores]
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            LOGGER.warning("[reranker] local cross-encoder unavailable; lexical fallback: %s", exc)
            scores = _lexical_scores(question, candidates)

    scored = [
        {**candidate, "score": float(score)}
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    return sorted(scored, key=lambda item: item["score"], reverse=True)

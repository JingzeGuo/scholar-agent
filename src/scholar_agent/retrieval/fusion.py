"""Explicit Reciprocal Rank Fusion (not hidden in a framework wrapper)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists with RRF.

    Parameters
    ----------
    rankings:
        Each inner sequence is a ranking best→worst of document/chunk IDs.
    k:
        RRF constant (default 60, as in Cormack et al.).

    Returns
    -------
    list of (id, fused_score) sorted by score descending, stable on ties by id.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def ranks_map(ranking: Sequence[str]) -> dict[str, int]:
    """Map id → 1-based rank (first occurrence wins)."""
    out: dict[str, int] = {}
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id not in out:
            out[doc_id] = rank
    return out

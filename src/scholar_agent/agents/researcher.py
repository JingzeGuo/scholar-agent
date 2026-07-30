"""Researcher Agent node."""

from __future__ import annotations

import logging
from collections.abc import Callable

from scholar_agent.config import Settings
from scholar_agent.models import AgentState
from scholar_agent.reranker import rerank
from scholar_agent.retrieval import RetrievalEngine, reciprocal_rank_fusion

LOGGER = logging.getLogger(__name__)
RerankFunction = Callable[[str, list[dict], str], list[dict]]


def researcher_node(
    state: AgentState,
    engine: RetrievalEngine,
    settings: Settings,
    rerank_function: RerankFunction = rerank,
) -> dict:
    """Run BM25, dense, graph, RRF, and reranking in a visible straight line."""
    queries = list(state["queries"])
    if state["feedback"]:
        queries.append(state["feedback"])

    sparse = engine.sparse_search(queries)
    dense = engine.dense_search(queries)
    graph = engine.graph_search(state["entities"])
    LOGGER.info(
        "[researcher] sparse=%d dense=%d graph=%d",
        len(sparse),
        len(dense),
        len(graph),
    )

    candidates = reciprocal_rank_fusion(sparse, dense, graph)
    LOGGER.info("[fusion] %d unique candidates", len(candidates))
    evidence = rerank_function(
        state["question"],
        candidates[:30],
        settings.reranker_model,
    )[:8]
    LOGGER.info("[reranker] selected %d evidence chunks", len(evidence))
    for index, item in enumerate(evidence, start=1):
        LOGGER.info(
            "[reranker] E%d %s p.%d score=%.3f",
            index,
            item["paper"],
            item["page"],
            item["score"],
        )

    retry_count = state["retry_count"] + (1 if state["feedback"] else 0)
    return {
        "candidates": candidates,
        "evidence": evidence,
        "retry_count": retry_count,
    }

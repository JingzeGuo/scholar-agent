"""Deterministic retrieval, fusion, reranking, and evidence-selection node."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable

from scholar_agent.agents.planner import target_matches
from scholar_agent.config import Settings
from scholar_agent.models import AgentState
from scholar_agent.reranker import rerank
from scholar_agent.retrieval import RetrievalEngine, reciprocal_rank_fusion

LOGGER = logging.getLogger(__name__)
RerankFunction = Callable[[list[str], list[dict], str], list[dict]]
MAX_EVIDENCE = 8
MAX_RERANK_CANDIDATES = 30
PER_QUERY_CANDIDATES = 8
PER_TARGET = 2
PER_PAPER = 4


def _select_evidence(
    items: list[dict],
    targets: list[str],
) -> list[dict]:
    ranked = sorted(
        items,
        key=lambda item: item["score"],
        reverse=True,
    )

    if targets and not any(
        target_matches(target, item["text"]) for target in targets for item in ranked
    ):
        return []

    selected: list[dict] = []
    selected_ids: set[str] = set()
    selected_pages: set[tuple[str, int]] = set()
    paper_counts: Counter = Counter()

    def add(item: dict) -> bool:
        page_key = (item["paper"], item["page"])

        if item["chunk_id"] in selected_ids:
            return False
        if page_key in selected_pages:
            return False
        if paper_counts[item["paper"]] >= PER_PAPER:
            return False

        selected.append(item)
        selected_ids.add(item["chunk_id"])
        selected_pages.add(page_key)
        paper_counts[item["paper"]] += 1
        return True

    # Give explicitly named targets a fair chance.
    for target in targets:
        added = 0
        for item in ranked:
            if target_matches(target, item["text"]) and add(item):
                added += 1
            if added >= PER_TARGET:
                break

    # Fill the remaining evidence slots purely by relevance.
    for item in ranked:
        add(item)
        if len(selected) >= MAX_EVIDENCE:
            break

    return selected


def _query_rankings(
    engine: RetrievalEngine,
    queries: list[str],
) -> tuple[list[list[dict]], list[list[dict]]]:
    sparse = [engine.sparse_search([query])[:PER_QUERY_CANDIDATES] for query in queries]
    dense = [
        ranking[:PER_QUERY_CANDIDATES] for ranking in engine.dense_search_many(queries)
    ]
    return sparse, dense


def _unique_count(rankings: list[list[dict]]) -> int:
    return len({item["chunk_id"] for ranking in rankings for item in ranking})


def researcher_node(
    state: AgentState,
    engine: RetrievalEngine,
    settings: Settings,
    rerank_function: RerankFunction = rerank,
) -> dict:
    """Run fixed hybrid retrieval, RRF, reranking, and evidence selection."""
    plan = state["plan"]
    corrective_query = state["verification"].get("corrective_query", "")
    queries = [corrective_query] if corrective_query else list(plan["queries"])

    sparse_rankings, dense_rankings = _query_rankings(engine, queries)
    LOGGER.info(
        "[researcher] queries=%d sparse_candidates=%d dense_candidates=%d",
        len(queries),
        _unique_count(sparse_rankings),
        _unique_count(dense_rankings),
    )

    candidates = reciprocal_rank_fusion(
        *(ranking for pair in zip(sparse_rankings, dense_rankings, strict=True) for ranking in pair),
    )
    candidates = candidates[:MAX_RERANK_CANDIDATES]
    LOGGER.info("[fusion] %d candidates for reranking", len(candidates))
    reranked = rerank_function(
        queries,
        candidates,
        settings.reranker_model,
    )
    retained = [item for item in reranked if item["score"] >= settings.min_rerank_score]
    LOGGER.info(
        "[reranker] retained=%d rejected=%d threshold=%.3f",
        len(retained),
        len(reranked) - len(retained),
        settings.min_rerank_score,
    )
    by_id = {item["chunk_id"]: item for item in state["evidence"]}
    for item in retained:
        previous = by_id.get(item["chunk_id"])
        if previous is None or item["score"] > previous["score"]:
            by_id[item["chunk_id"]] = item
    eligible = [item for item in by_id.values() if item["score"] >= settings.min_rerank_score]
    evidence = _select_evidence(
        eligible,
        plan["targets"],
    )
    LOGGER.info("[reranker] selected %d evidence chunks", len(evidence))
    for index, item in enumerate(evidence, start=1):
        LOGGER.info(
            "[reranker] E%d %s p.%d score=%.3f",
            index,
            item["paper"],
            item["page"],
            item["score"],
        )

    retry_count = state["retry_count"] + (1 if corrective_query else 0)
    stop_reason = "" if evidence else "no_relevant_evidence"
    if corrective_query and {item["chunk_id"] for item in evidence} == {
        item["chunk_id"] for item in state["evidence"]
    }:
        evidence = state["evidence"]
        stop_reason = "no_new_evidence"
        LOGGER.info("[researcher] retry produced no new evidence")
    return {
        "evidence": evidence,
        "retry_count": retry_count,
        "stop_reason": stop_reason,
    }

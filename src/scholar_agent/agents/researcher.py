"""Researcher Agent node."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable

from scholar_agent.agents.planner import target_matches
from scholar_agent.config import Settings
from scholar_agent.graph_store import extract_entities
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


def _paper_key(paper: str) -> tuple[int, int, str]:
    match = re.match(r"(\d{4})\.(\d+)\.pdf$", paper)
    return (int(match.group(1)), int(match.group(2)), paper) if match else (9999, 99999, paper)


def _target_ratios(targets: list[str], chunks: list[dict]) -> dict[str, dict[str, float]]:
    totals = Counter(item["paper"] for item in chunks)
    hits: dict[str, Counter] = {target: Counter() for target in targets}
    for item in chunks:
        for target in targets:
            if target_matches(target, item["text"]):
                hits[target][item["paper"]] += 1
    return {
        target: {paper: count / totals[paper] for paper, count in target_hits.items()}
        for target, target_hits in hits.items()
    }


def _select_evidence(items: list[dict], targets: list[str], chunks: list[dict]) -> list[dict]:
    ratios = _target_ratios(targets, chunks)
    available_targets = [target for target in targets if ratios[target]]
    if targets and not available_targets:
        return []
    selected: list[dict] = []
    selected_ids: set[str] = set()
    selected_pages: set[tuple[str, int]] = set()
    paper_counts: Counter = Counter()

    def add(
        item: dict,
        enforce_paper_cap: bool = True,
        enforce_unique_page: bool = True,
    ) -> bool:
        page = (item["paper"], item["page"])
        if item["chunk_id"] in selected_ids:
            return False
        if enforce_unique_page and page in selected_pages:
            return False
        if enforce_paper_cap and paper_counts[item["paper"]] >= PER_PAPER:
            return False
        selected.append(item)
        selected_ids.add(item["chunk_id"])
        selected_pages.add(page)
        paper_counts[item["paper"]] += 1
        return True

    for target in available_targets:
        matches = [item for item in items if target_matches(target, item["text"])]
        primary_paper = min({item["paper"] for item in matches}, key=_paper_key, default="")
        matches.sort(
            key=lambda item: (
                item["paper"] == primary_paper,
                ratios[target].get(item["paper"], 0.0),
                item["score"],
            ),
            reverse=True,
        )
        added = 0
        for unique_page in (True, False):
            for item in matches:
                if add(item, enforce_unique_page=unique_page):
                    added += 1
                if added == PER_TARGET:
                    break
            if added == PER_TARGET:
                break

    ranked = sorted(items, key=lambda item: item["score"], reverse=True)
    for enforce_cap, unique_page in ((True, True), (False, True), (True, False), (False, False)):
        for item in ranked:
            add(item, enforce_cap, unique_page)
            if len(selected) == MAX_EVIDENCE:
                return selected
    return selected


def _query_rankings(
    engine: RetrievalEngine,
    queries: list[str],
    fallback_entities: list[str],
) -> tuple[list[list[dict]], list[list[dict]], list[list[dict]]]:
    sparse: list[list[dict]] = []
    dense: list[list[dict]] = []
    graph: list[list[dict]] = []
    for query in queries:
        sparse.append(engine.sparse_search([query])[:PER_QUERY_CANDIDATES])
        dense.append(engine.dense_search([query])[:PER_QUERY_CANDIDATES])
        entities = extract_entities(query) or fallback_entities
        graph.append(engine.graph_search(entities)[:PER_QUERY_CANDIDATES])
    return sparse, dense, graph


def _unique_count(rankings: list[list[dict]]) -> int:
    return len({item["chunk_id"] for ranking in rankings for item in ranking})


def researcher_node(
    state: AgentState,
    engine: RetrievalEngine,
    settings: Settings,
    rerank_function: RerankFunction = rerank,
) -> dict:
    """Run BM25, dense, graph, RRF, and reranking in a visible straight line."""
    plan = state["plan"]
    queries = list(plan["queries"])
    corrective_query = state["verification"].get("corrective_query", "")
    if corrective_query:
        queries.append(corrective_query)

    sparse, dense, graph = _query_rankings(engine, queries, plan["entities"])
    LOGGER.info(
        "[researcher] sparse=%d dense=%d graph=%d",
        _unique_count(sparse),
        _unique_count(dense),
        _unique_count(graph),
    )

    candidates = reciprocal_rank_fusion(*sparse, *dense, *graph)
    LOGGER.info("[fusion] %d unique candidates", len(candidates))
    reranked = rerank_function(
        queries,
        candidates[:MAX_RERANK_CANDIDATES],
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
    evidence = _select_evidence(eligible, plan["targets"], engine.chunks)
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

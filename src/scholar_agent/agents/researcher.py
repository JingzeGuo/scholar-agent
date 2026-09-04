"""Deterministic retrieval, fusion, reranking, and evidence-selection node."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable

from scholar_agent.agents.planner import requirement_targets, target_matches
from scholar_agent.config import Settings
from scholar_agent.models import AgentState
from scholar_agent.reranker import rerank
from scholar_agent.retrieval import RetrievalEngine, reciprocal_rank_fusion

LOGGER = logging.getLogger(__name__)
RerankFunction = Callable[[list[str], list[dict], str], list[dict]]
MAX_EVIDENCE = 8
MAX_RERANK_CANDIDATES = 30
PER_QUERY_RERANK_CANDIDATES = 4
PER_TARGET = 2
PER_PAPER = 4


def _select_evidence(
    items: list[dict],
    requirements: list[dict],
    min_score: float = float("-inf"),
) -> list[dict]:
    targets = requirement_targets(requirements)
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

        if len(selected) >= MAX_EVIDENCE:
            return False
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

    def requirement_score(item: dict, requirement_id: str) -> float:
        scores = item.get("_requirement_scores", {})
        if isinstance(scores, dict):
            score = scores.get(requirement_id)
            if isinstance(score, int | float):
                return float(score)
        return float(item["score"])

    def matches_requirement(item: dict, requirement: dict) -> bool:
        return not requirement["targets"] or any(
            target_matches(target, item["text"]) for target in requirement["targets"]
        )

    requirement_rankings: dict[str, list[dict]] = {
        requirement["id"]: sorted(
            items,
            key=lambda item: requirement_score(item, requirement["id"]),
            reverse=True,
        )
        for requirement in requirements
    }

    # Reserve one relevant evidence slot for every independently verified requirement.
    for requirement in requirements:
        for item in requirement_rankings[requirement["id"]]:
            if requirement_score(item, requirement["id"]) < min_score:
                break
            if not matches_requirement(item, requirement):
                continue
            if item["chunk_id"] in selected_ids or add(item):
                break

    # A comparison requirement may need separate evidence for each named target.
    for requirement in requirements:
        for target in requirement["targets"]:
            if any(
                target_matches(target, item["text"])
                and requirement_score(item, requirement["id"]) >= min_score
                for item in selected
            ):
                continue
            for item in requirement_rankings[requirement["id"]]:
                if requirement_score(item, requirement["id"]) < min_score:
                    break
                if target_matches(target, item["text"]) and add(item):
                    break

    # Preserve the existing per-target diversity after requirement coverage.
    for target in targets:
        added = sum(target_matches(target, item["text"]) for item in selected)
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


def _planned_queries(plan: dict) -> tuple[list[str], list[list[str]]]:
    queries: list[str] = []
    query_requirement_ids: list[list[str]] = []
    for requirement in plan["requirements"]:
        queries.append(requirement["query"].strip())
        query_requirement_ids.append([requirement["id"]])
    return queries, query_requirement_ids


def _attach_requirement_scores(
    items: list[dict],
    query_requirement_ids: list[list[str]],
) -> list[dict]:
    scored: list[dict] = []
    for item in items:
        raw_query_scores = item.get("_query_scores")
        if not isinstance(raw_query_scores, list) or len(raw_query_scores) != len(
            query_requirement_ids,
        ):
            raw_query_scores = [item["score"]] * len(query_requirement_ids)

        requirement_scores: dict[str, float] = {}
        for score, requirement_ids in zip(
            raw_query_scores,
            query_requirement_ids,
            strict=True,
        ):
            for requirement_id in requirement_ids:
                requirement_scores[requirement_id] = max(
                    float(score),
                    requirement_scores.get(requirement_id, float("-inf")),
                )

        clean_item = {key: value for key, value in item.items() if key != "_query_scores"}
        clean_item["_requirement_scores"] = requirement_scores
        scored.append(clean_item)
    return scored


def _merge_evidence(existing: list[dict], new: list[dict]) -> list[dict]:
    by_id = {item["chunk_id"]: dict(item) for item in existing}
    for item in new:
        previous = by_id.get(item["chunk_id"])
        if previous is None:
            by_id[item["chunk_id"]] = item
            continue

        winner = item if item["score"] > previous["score"] else previous
        merged = dict(winner)
        merged["score"] = max(float(previous["score"]), float(item["score"]))
        requirement_scores: dict[str, float] = {}
        for source in (previous, item):
            raw_scores = source.get("_requirement_scores", {})
            if not isinstance(raw_scores, dict):
                continue
            for requirement_id, score in raw_scores.items():
                if isinstance(requirement_id, str) and isinstance(score, int | float):
                    requirement_scores[requirement_id] = max(
                        float(score),
                        requirement_scores.get(requirement_id, float("-inf")),
                    )
        merged["_requirement_scores"] = requirement_scores
        by_id[item["chunk_id"]] = merged
    return list(by_id.values())


def _query_rankings(
    engine: RetrievalEngine,
    queries: list[str],
) -> tuple[list[list[dict]], list[list[dict]]]:
    sparse = [engine.sparse_search([query]) for query in queries]
    dense = engine.dense_search_many(queries)
    return sparse, dense


def _unique_count(rankings: list[list[dict]]) -> int:
    return len({item["chunk_id"] for ranking in rankings for item in ranking})


def _rerank_candidates(
    sparse_rankings: list[list[dict]],
    dense_rankings: list[list[dict]],
) -> list[dict]:
    global_ranking = reciprocal_rank_fusion(
        *(
            ranking
            for pair in zip(sparse_rankings, dense_rankings, strict=True)
            for ranking in pair
        ),
    )
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add(item: dict) -> None:
        if len(selected) < MAX_RERANK_CANDIDATES and item["chunk_id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["chunk_id"])

    # Keep local candidates before global fusion can favor evidence repeated by other queries.
    for sparse, dense in zip(sparse_rankings, dense_rankings, strict=True):
        for item in reciprocal_rank_fusion(sparse, dense)[:PER_QUERY_RERANK_CANDIDATES]:
            add(item)

    for item in global_ranking:
        add(item)
    return selected


def researcher_node(
    state: AgentState,
    engine: RetrievalEngine,
    settings: Settings,
    rerank_function: RerankFunction = rerank,
) -> dict:
    """Run fixed hybrid retrieval, RRF, reranking, and evidence selection."""
    plan = state["plan"]
    corrective_query = state["verification"].get("corrective_query", "")
    if corrective_query:
        requirement_ids = state["verification"].get("missing") or [
            requirement["id"] for requirement in plan["requirements"]
        ]
        queries = [corrective_query]
        query_requirement_ids = [requirement_ids]
    else:
        queries, query_requirement_ids = _planned_queries(plan)

    sparse_rankings, dense_rankings = _query_rankings(engine, queries)
    LOGGER.info(
        "[researcher] queries=%d sparse_candidates=%d dense_candidates=%d",
        len(queries),
        _unique_count(sparse_rankings),
        _unique_count(dense_rankings),
    )

    candidates = _rerank_candidates(sparse_rankings, dense_rankings)
    LOGGER.info("[fusion] %d candidates for reranking", len(candidates))
    reranked = rerank_function(
        queries,
        candidates,
        settings.reranker_model,
    )
    reranked = _attach_requirement_scores(reranked, query_requirement_ids)
    retained = [item for item in reranked if item["score"] >= settings.min_rerank_score]
    LOGGER.info(
        "[reranker] retained=%d rejected=%d threshold=%.3f",
        len(retained),
        len(reranked) - len(retained),
        settings.min_rerank_score,
    )
    merged = _merge_evidence(state["evidence"], retained)
    eligible = [item for item in merged if item["score"] >= settings.min_rerank_score]
    evidence = _select_evidence(
        eligible,
        plan["requirements"],
        settings.min_rerank_score,
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

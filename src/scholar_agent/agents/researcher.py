"""Deterministic retrieval, fusion, reranking, and evidence-selection node."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable

from scholar_agent.agents.planner import (
    DEFAULT_TOP_K,
    MAX_REQUIREMENTS,
    MAX_TARGETS_PER_REQUIREMENT,
    RETRIEVAL_STRATEGIES,
    evidence_matches_target,
    requirement_targets,
    sanitize_top_k,
)
from scholar_agent.config import Settings
from scholar_agent.models import AgentState
from scholar_agent.reranker import rerank
from scholar_agent.retrieval import RetrievalEngine, reciprocal_rank_fusion

LOGGER = logging.getLogger(__name__)
RerankFunction = Callable[[list[str], list[dict], str], list[dict]]
DEFAULT_EVIDENCE_LIMIT = 8
MAX_EVIDENCE = MAX_REQUIREMENTS * MAX_TARGETS_PER_REQUIREMENT
MAX_RERANK_CANDIDATES = 30
PER_QUERY_RERANK_CANDIDATES = 4
PER_TARGET = 2
PER_PAPER = 4


def _requirement_score(item: dict, requirement_id: str) -> float:
    scores = item.get("_requirement_scores")
    if isinstance(scores, dict):
        score = scores.get(requirement_id)
        if isinstance(score, int | float):
            return float(score)
        return float("-inf")
    return float(item["score"])


def _build_evidence_board(
    items: list[dict],
    requirements: list[dict],
    min_score: float,
) -> tuple[list[dict], dict[str, dict]]:
    """Link selected passages to requirements without changing selection or citation order."""
    board = {
        requirement["id"]: {"requirement": requirement["description"], "evidence_ids": []}
        for requirement in requirements
    }
    evidence = []
    for index, item in enumerate(items, start=1):
        evidence_id = f"E{index}"
        supports = [
            requirement["id"]
            for requirement in requirements
            if _requirement_score(item, requirement["id"]) >= min_score
        ]
        evidence.append(
            {
                **{key: value for key, value in item.items() if key != "_requirement_scores"},
                "id": evidence_id,
                "paper_id": item["paper"],
                "title": item.get("title"),
                "section": item.get("section"),
                "supports": supports,
                "requirement_scores": dict(item["_requirement_scores"]),
            },
        )
        for requirement_id in supports:
            board[requirement_id]["evidence_ids"].append(evidence_id)
    evidence_by_id = {item["id"]: item for item in evidence}
    for requirement_id, entry in board.items():
        entry["evidence_ids"].sort(
            key=lambda evidence_id: evidence_by_id[evidence_id]["requirement_scores"][requirement_id],
            reverse=True,
        )
    return evidence, board


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

    selected: list[dict] = []
    selected_ids: set[str] = set()
    selected_pages: set[tuple[str, int]] = set()
    paper_counts: Counter = Counter()

    def add(item: dict, *, enforce_diversity: bool, limit: int) -> bool:
        page_key = (item["paper"], item["page"])

        if len(selected) >= limit:
            return False
        if item["chunk_id"] in selected_ids:
            return False
        if enforce_diversity:
            if page_key in selected_pages:
                return False
            if paper_counts[item["paper"]] >= PER_PAPER:
                return False

        selected.append(item)
        selected_ids.add(item["chunk_id"])
        selected_pages.add(page_key)
        paper_counts[item["paper"]] += 1
        return True

    def matches_requirement(item: dict, requirement: dict) -> bool:
        return not requirement["targets"] or any(
            evidence_matches_target(target, item) for target in requirement["targets"]
        )

    requirement_rankings: dict[str, list[dict]] = {
        requirement["id"]: sorted(
            items,
            key=lambda item: _requirement_score(item, requirement["id"]),
            reverse=True,
        )
        for requirement in requirements
    }

    # Reserve one relevant evidence slot for every planned requirement.
    for requirement in requirements:
        for item in requirement_rankings[requirement["id"]]:
            if _requirement_score(item, requirement["id"]) < min_score:
                break
            if not matches_requirement(item, requirement):
                continue
            if item["chunk_id"] in selected_ids or add(
                item,
                enforce_diversity=False,
                limit=MAX_EVIDENCE,
            ):
                break

    # A comparison requirement may need separate evidence for each named target.
    for requirement in requirements:
        for target in requirement["targets"]:
            if any(
                evidence_matches_target(target, item)
                and _requirement_score(item, requirement["id"]) >= min_score
                for item in selected
            ):
                continue
            for item in requirement_rankings[requirement["id"]]:
                if _requirement_score(item, requirement["id"]) < min_score:
                    break
                if evidence_matches_target(target, item) and add(
                    item,
                    enforce_diversity=False,
                    limit=MAX_EVIDENCE,
                ):
                    break

    evidence_limit = max(DEFAULT_EVIDENCE_LIMIT, len(selected))

    # Preserve the existing per-target diversity after requirement coverage.
    for target in targets:
        added = sum(evidence_matches_target(target, item) for item in selected)
        for item in ranked:
            if evidence_matches_target(target, item) and add(
                item,
                enforce_diversity=True,
                limit=evidence_limit,
            ):
                added += 1
            if added >= PER_TARGET:
                break

    # Fill the remaining evidence slots purely by relevance.
    for item in ranked:
        add(item, enforce_diversity=True, limit=evidence_limit)
        if len(selected) >= evidence_limit:
            break

    return selected


def _planned_queries(plan: dict) -> tuple[list[str], list[list[str]]]:
    queries: list[str] = []
    query_requirement_ids: list[list[str]] = []
    for requirement in plan["requirements"]:
        queries.append(requirement["query"].strip())
        query_requirement_ids.append([requirement["id"]])
    return queries, query_requirement_ids


def _retrieval_requests(plan: dict, retrieval_mode: str) -> list[dict]:
    if retrieval_mode not in {"fixed_hybrid", "adaptive"}:
        raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")

    requests: list[dict] = []
    for requirement in plan["requirements"]:
        planned_strategy = requirement.get("retrieval_strategy")
        strategy = (
            planned_strategy
            if retrieval_mode == "adaptive" and planned_strategy in RETRIEVAL_STRATEGIES
            else "hybrid"
        )
        requests.append(
            {
                "requirement_id": requirement["id"],
                "query": requirement["query"].strip(),
                "retrieval_strategy": strategy,
                "top_k": sanitize_top_k(requirement.get("top_k", DEFAULT_TOP_K)),
            },
        )
    return requests


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
            raise ValueError("Reranker must return one query score per query")

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


def _execute_retrieval(
    engine: RetrievalEngine,
    requests: list[dict],
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Return one effective ranking per requirement plus raw rankings for global RRF."""
    sparse: dict[int, list[dict]] = {}
    dense: dict[int, list[dict]] = {}

    for index, request in enumerate(requests):
        if request["retrieval_strategy"] in {"bm25", "hybrid"}:
            sparse[index] = engine.sparse_search(
                [request["query"]],
                top_k=request["top_k"],
            )

    dense_groups: dict[int, list[tuple[int, str]]] = {}
    for index, request in enumerate(requests):
        if request["retrieval_strategy"] in {"dense", "hybrid"}:
            dense_groups.setdefault(request["top_k"], []).append(
                (index, request["query"]),
            )
    for top_k, group in dense_groups.items():
        rankings = engine.dense_search_many(
            [query for _, query in group],
            top_k=top_k,
        )
        if len(rankings) != len(group):
            raise ValueError("Dense retrieval must return one ranking per query")
        for (index, _), ranking in zip(group, rankings, strict=True):
            dense[index] = ranking

    effective_rankings: list[list[dict]] = []
    source_rankings: list[list[dict]] = []
    for index, request in enumerate(requests):
        strategy = request["retrieval_strategy"]
        if strategy == "bm25":
            ranking = sparse[index]
            sources = [ranking]
        elif strategy == "dense":
            ranking = dense[index]
            sources = [ranking]
        else:
            sources = [sparse[index], dense[index]]
            ranking = reciprocal_rank_fusion(*sources)
        effective_rankings.append(ranking)
        source_rankings.extend(sources)
    return effective_rankings, source_rankings


def _unique_count(rankings: list[list[dict]]) -> int:
    return len({item["chunk_id"] for ranking in rankings for item in ranking})


def _page_refs(items: Iterable[dict]) -> list[dict]:
    return [
        {"paper": paper, "page": page}
        for paper, page in sorted({(item["paper"], item["page"]) for item in items})
    ]


def _select_candidates_for_reranking(
    requirement_rankings: list[list[dict]],
    source_rankings: list[list[dict]] | None = None,
) -> list[dict]:
    global_ranking = reciprocal_rank_fusion(*(source_rankings or requirement_rankings))
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add(item: dict) -> bool:
        if len(selected) < MAX_RERANK_CANDIDATES and item["chunk_id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["chunk_id"])
            return True
        return False

    # Keep local candidates before global fusion can favor evidence repeated by other queries.
    for ranking in requirement_rankings:
        added = 0
        for item in ranking:
            if add(item):
                added += 1
            if added >= PER_QUERY_RERANK_CANDIDATES:
                break

    for item in global_ranking:
        add(item)
    return selected


def researcher_node(
    state: AgentState,
    engine: RetrievalEngine,
    settings: Settings,
    rerank_function: RerankFunction = rerank,
) -> dict:
    """Execute each planned retrieval strategy, then rerank and select evidence."""
    plan = state["plan"]
    queries, query_requirement_ids = _planned_queries(plan)
    requests = _retrieval_requests(plan, state["retrieval_mode"])
    requirement_rankings, source_rankings = _execute_retrieval(engine, requests)
    sparse_rankings = [
        ranking
        for request, ranking in zip(requests, requirement_rankings, strict=True)
        if request["retrieval_strategy"] == "bm25"
    ]
    dense_rankings = [
        ranking
        for request, ranking in zip(requests, requirement_rankings, strict=True)
        if request["retrieval_strategy"] == "dense"
    ]
    LOGGER.info(
        "[researcher] mode=%s queries=%d single_bm25=%d single_dense=%d hybrid=%d "
        "candidates=%d",
        state["retrieval_mode"],
        len(queries),
        len(sparse_rankings),
        len(dense_rankings),
        sum(request["retrieval_strategy"] == "hybrid" for request in requests),
        _unique_count(source_rankings),
    )

    candidates = _select_candidates_for_reranking(
        requirement_rankings,
        source_rankings,
    )
    retrieval_stages = {
        "retrieval": _page_refs(item for ranking in source_rankings for item in ranking),
        "rerank": _page_refs(candidates),
    }
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
    evidence = _select_evidence(
        retained,
        plan["requirements"],
        settings.min_rerank_score,
    )
    evidence, evidence_board = _build_evidence_board(
        evidence,
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

    return {
        "evidence": evidence,
        "evidence_board": evidence_board,
        "retrieval_trace": requests,
        "retrieval_stages": retrieval_stages,
    }

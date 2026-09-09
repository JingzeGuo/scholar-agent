"""Execute one bounded batch of Controller-selected recovery actions."""

from __future__ import annotations

from collections.abc import Callable

from scholar_agent.agents.planner import MAX_TOP_K, evidence_matches_target
from scholar_agent.agents.researcher import (
    _attach_requirement_scores,
    _build_evidence_board,
    _execute_retrieval,
    _retrieval_requests,
    _select_candidates_for_reranking,
    recovery_trace_entry,
)
from scholar_agent.config import Settings
from scholar_agent.models import AgentState
from scholar_agent.reranker import rerank
from scholar_agent.retrieval import PAPER_SEARCH_RESULTS, RetrievalEngine

RerankFunction = Callable[[list[str], list[dict], str], list[dict]]
PER_REQUIREMENT_EVIDENCE = 2
MAX_RECOVERY_EVIDENCE = 4


def _raw_evidence(item: dict) -> dict:
    return {
        **{
            key: value
            for key, value in item.items()
            if key not in {"id", "paper_id", "supports", "requirement_scores"}
        },
        "_requirement_scores": dict(item["requirement_scores"]),
    }


def _page_refs(items: list[dict]) -> list[dict]:
    return [
        {"paper": paper, "page": page}
        for paper, page in sorted({(item["paper"], item["page"]) for item in items})
    ]


def _candidates(
    action: dict,
    requirement: dict,
    state: AgentState,
    engine: RetrievalEngine,
) -> tuple[list[dict], dict]:
    if action["action"] == "search_within_paper":
        parameters = {
            "paper": action["paper"],
            "query": action["query"],
            "top_k": PAPER_SEARCH_RESULTS,
        }
        return engine.search_within_paper(**parameters), parameters
    if action["action"] == "expand_neighbors":
        parameters = {
            "chunk_id": action["chunk_id"],
            "radius": 1,
            "query": action["query"],
        }
        return engine.expand_neighbors(action["chunk_id"], radius=1), parameters

    recovered = {**requirement, "query": action["query"], "top_k": MAX_TOP_K}
    requests = _retrieval_requests({"requirements": [recovered]}, state["retrieval_mode"])
    rankings, sources = _execute_retrieval(engine, requests)
    request = requests[0]
    parameters = {**request, "from_top_k": requirement["top_k"]}
    return _select_candidates_for_reranking(rankings, sources), parameters


def recovery_node(
    state: AgentState,
    engine: RetrievalEngine,
    settings: Settings,
    rerank_function: RerankFunction = rerank,
) -> dict:
    """Execute sanitized actions, rerank their results, and append bounded evidence."""
    requirements = {item["id"]: item for item in state["plan"]["requirements"]}
    action_rankings = []
    traces = []
    all_candidates = []
    for action in state["controller_trace"]["actions"]:
        requirement = requirements[action["requirement_id"]]
        candidates, parameters = _candidates(action, requirement, state, engine)
        ranked = rerank_function([action["query"]], candidates, settings.reranker_model)
        ranked = _attach_requirement_scores(ranked, [[requirement["id"]]])
        action_rankings.append((action, requirement, ranked))
        all_candidates.extend(candidates)
        trace = recovery_trace_entry(
            requirement["id"], action["action"], "controller_evidence_gap", 1,
            parameters, candidates, ranked,
        )
        if action.get("reason"):
            trace["controller_reason"] = action["reason"]
        traces.append(trace)

    merged: dict[str, dict] = {}
    for _, _, ranking in action_rankings:
        for item in ranking:
            current = merged.get(item["chunk_id"])
            if current is None:
                merged[item["chunk_id"]] = item
                continue
            current["score"] = max(current["score"], item["score"])
            for requirement_id, score in item["_requirement_scores"].items():
                current["_requirement_scores"][requirement_id] = max(
                    score,
                    current["_requirement_scores"].get(requirement_id, float("-inf")),
                )

    existing_ids = {item["chunk_id"] for item in state["evidence"]}
    added_ids = []
    for action, requirement, ranking in action_rankings:
        added = 0
        for item in ranking:
            if item["score"] < settings.min_rerank_score:
                continue
            if action["action"] == "increase_depth" and requirement["targets"] and not any(
                evidence_matches_target(target, item) for target in requirement["targets"]
            ):
                continue
            chunk_id = item["chunk_id"]
            if chunk_id in existing_ids:
                continue
            if chunk_id not in added_ids:
                added_ids.append(chunk_id)
            added += 1
            if added >= PER_REQUIREMENT_EVIDENCE or len(added_ids) >= MAX_RECOVERY_EVIDENCE:
                break

    combined = [
        *map(_raw_evidence, state["evidence"]),
        *(merged[chunk_id] for chunk_id in added_ids),
    ]
    evidence, board = _build_evidence_board(
        combined, state["plan"]["requirements"], settings.min_rerank_score,
    )
    for requirement_id, entry in board.items():
        entry["candidate_papers"] = state["evidence_board"][requirement_id].get(
            "candidate_papers", [],
        )

    selected_ids = {item["chunk_id"] for item in evidence}
    for trace in traces:
        for result in trace["results"]:
            result["selected"] = result["chunk_id"] in selected_ids
            result["added"] = result["chunk_id"] in added_ids
    stages = dict(state["retrieval_stages"])
    stages["post_recovery"] = _page_refs(
        [*stages.get("retrieval", []), *all_candidates],
    )
    return {
        "evidence": evidence,
        "evidence_board": board,
        "recovery_trace": traces,
        "retrieval_stages": stages,
    }

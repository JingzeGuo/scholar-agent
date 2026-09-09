"""One-shot evidence-gap decisions over the Researcher's observed state."""

from __future__ import annotations

import logging

from scholar_agent.agents.planner import MAX_TOP_K
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
ACTIONS = frozenset({"search_within_paper", "expand_neighbors", "increase_depth"})
MAX_ACTIONS = 2


def _controller_prompt(state: AgentState) -> str:
    evidence = {item["id"]: item for item in state["evidence"]}
    blocks = []
    for requirement in state["plan"]["requirements"]:
        requirement_id = requirement["id"]
        board = state["evidence_board"][requirement_id]
        passages = []
        for evidence_id in board["evidence_ids"]:
            item = evidence[evidence_id]
            passages.append(
                f"[{evidence_id}] {item.get('title') or item['paper']} "
                f"({item['paper']}) p.{item['page']} score="
                f"{item['requirement_scores'][requirement_id]:.3f}\n{item['text']}",
            )
        papers = "\n".join(
            f"- {item.get('title') or 'Unknown title'} ({item['paper']}), "
            f"best_score={item['best_score']:.3f}, selected={item['selected']}"
            for item in board.get("candidate_papers", [])
        )
        blocks.append(
            f"Requirement {requirement_id}: {requirement['description']}\n"
            f"Targets: {requirement['targets']}\n"
            f"Initial query: {requirement['query']}\n"
            f"Strategy/top_k: {requirement['retrieval_strategy']}/{requirement['top_k']}\n"
            f"Observed candidate papers:\n{papers or 'None'}\n"
            f"Selected evidence:\n" + "\n\n".join(passages or ["None"]),
        )

    return f"""You are the Evidence-Gap Controller in an academic research workflow.
Inspect the first retrieval observation and decide whether one bounded follow-up action could
recover evidence missing from a requirement. Do not answer the question.

Return one JSON object with exactly one field, "actions", containing zero to {MAX_ACTIONS} objects.
An empty list means the current evidence is sufficient or no useful bounded action is available.
Use at most one action per requirement. Every action must contain a requirement_id, one action
from {sorted(ACTIONS)}, a concise evidence-seeking query, and a brief reason.

- search_within_paper: also provide a paper from that requirement's observed candidate papers.
- expand_neighbors: also provide an evidence chunk_id listed for that requirement.
- increase_depth: use only when its current top_k is below {MAX_TOP_K}; Python fixes top_k to
  {MAX_TOP_K}.

Base decisions only on the question, requirements, selected passages, scores, and candidate-paper
metadata below. A reason is for debugging only. Generate search terms that seek missing evidence;
do not state an answer or assume that a relevance score proves support.

Question: {state['question']}

{chr(10).join(blocks)}
"""


def sanitize_actions(payload: object, state: AgentState) -> tuple[list[dict], int]:
    raw_actions = payload.get("actions") if isinstance(payload, dict) else None
    if not isinstance(raw_actions, list):
        return [], 1

    requirements = {item["id"]: item for item in state["plan"]["requirements"]}
    actions = []
    used_requirements = set()
    rejected = 0
    for raw in raw_actions:
        if not isinstance(raw, dict):
            rejected += 1
            continue
        requirement_id = raw.get("requirement_id")
        action = raw.get("action")
        query = raw.get("query")
        if (
            len(actions) >= MAX_ACTIONS
            or requirement_id not in requirements
            or requirement_id in used_requirements
            or action not in ACTIONS
            or not isinstance(query, str)
            or not query.strip()
        ):
            rejected += 1
            continue

        board = state["evidence_board"][requirement_id]
        clean = {
            "requirement_id": requirement_id,
            "action": action,
            "query": query.strip(),
        }
        if action == "search_within_paper":
            paper = raw.get("paper")
            if paper not in {item["paper"] for item in board.get("candidate_papers", [])}:
                rejected += 1
                continue
            clean["paper"] = paper
        elif action == "expand_neighbors":
            chunk_id = raw.get("chunk_id")
            allowed_ids = {
                item["chunk_id"]
                for item in state["evidence"]
                if item["id"] in board["evidence_ids"]
            }
            if chunk_id not in allowed_ids:
                rejected += 1
                continue
            clean["chunk_id"] = chunk_id
        elif requirements[requirement_id]["top_k"] >= MAX_TOP_K:
            rejected += 1
            continue

        reason = raw.get("reason")
        if isinstance(reason, str) and reason.strip():
            clean["reason"] = reason.strip()
        actions.append(clean)
        used_requirements.add(requirement_id)
    return actions, rejected


def controller_node(state: AgentState, llm: LLMClient) -> dict:
    """Choose zero to two valid actions from one retrieval observation."""
    try:
        payload = llm.complete_json(_controller_prompt(state))
    except ValueError:
        LOGGER.warning("[controller] invalid JSON; continuing without recovery")
        payload = None
    actions, rejected = sanitize_actions(payload, state)
    LOGGER.info("[controller] actions=%d rejected=%d", len(actions), rejected)
    return {"controller_trace": {"actions": actions, "rejected_actions": rejected}}

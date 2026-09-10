"""One-shot evidence-gap decisions over the Researcher's observed state."""

from __future__ import annotations

import logging
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from scholar_agent.agents.planner import MAX_TOP_K
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
ACTIONS = frozenset({"search_within_paper", "expand_neighbors", "increase_depth"})
MAX_ACTIONS = 2
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _ControllerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchWithinPaper(_ControllerModel):
    tool: Literal["search_within_paper"]
    candidate_id: NonEmptyString
    query: NonEmptyString


class ExpandNeighbors(_ControllerModel):
    tool: Literal["expand_neighbors"]
    chunk_id: NonEmptyString
    query: NonEmptyString


class IncreaseDepth(_ControllerModel):
    tool: Literal["increase_depth"]
    query: NonEmptyString


RecoveryAction = Annotated[
    SearchWithinPaper | ExpandNeighbors | IncreaseDepth,
    Field(discriminator="tool"),
]


class RequirementAssessment(_ControllerModel):
    requirement_id: NonEmptyString
    status: Literal["sufficient", "missing", "unresolved"]
    covered: list[NonEmptyString]
    missing: list[NonEmptyString]
    action: RecoveryAction | None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status == "sufficient" and (self.missing or self.action):
            raise ValueError("sufficient requirements cannot have missing aspects or actions")
        if self.status == "missing" and (not self.missing or not self.action):
            raise ValueError("missing requirements need missing aspects and an action")
        if self.status == "unresolved" and (not self.missing or self.action):
            raise ValueError("unresolved requirements need missing aspects and no action")
        return self


class ControllerPayload(_ControllerModel):
    assessments: list[object]


def _rejection(raw: object, reason: str) -> dict:
    action = raw if isinstance(raw, dict) else {}
    nested = action.get("action")
    selector = nested if isinstance(nested, dict) else action
    rejection = {
        "requirement_id": action.get("requirement_id"),
        "action": nested.get("tool") if isinstance(nested, dict) else nested,
        "reason": reason,
    }
    for key in ("candidate_id", "paper", "chunk_id"):
        if key in selector:
            rejection[key] = selector[key]
    return rejection


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
                f"[{evidence_id}] chunk_id={item['chunk_id']} | "
                f"{item.get('title') or item['paper']} "
                f"({item['paper']}) p.{item['page']} score="
                f"{item['requirement_scores'][requirement_id]:.3f}\n{item['text']}",
            )
        papers = "\n".join(
            f"- [P{index}] {item.get('title') or 'Unknown title'} ({item['paper']}), "
            f"best_score={item['best_score']:.3f}, selected={item['selected']}"
            for index, item in enumerate(board.get("candidate_papers", []), start=1)
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

Return one JSON object with exactly one field, "assessments", containing one object per requirement.
Each assessment must contain exactly: requirement_id, status, covered, missing, and action.
"covered" and "missing" are lists of concise, distinct aspects from the requirement. Use these statuses:
- sufficient: all explicitly requested aspects have direct support; missing is empty and action is null.
- missing: support is missing and one bounded follow-up may help; missing and action are required.
- unresolved: support is missing but no displayed bounded action can target it; action is null.

Across all assessments, use at most {MAX_ACTIONS} non-null actions and at most one per requirement.
Each action contains a tool from {sorted(ACTIONS)} and a concise evidence-seeking query.

- search_within_paper: provide `"tool": "search_within_paper"` and `"candidate_id": "P1"` by
  copying a displayed [P#] ID for that
  requirement; do not return a title or filename.
- expand_neighbors: provide `"tool": "expand_neighbors"` and copy the exact evidence chunk_id
  shown for that requirement.
- increase_depth: provide `"tool": "increase_depth"`; use it only when the current top_k is below
  {MAX_TOP_K}. Python fixes top_k to {MAX_TOP_K}.

Base decisions only on the question, requirements, selected passages, scores, and candidate-paper
metadata below. Generate search terms that seek missing evidence; do not state an answer or assume
that a relevance score proves support.

Question: {state["question"]}

{chr(10).join(blocks)}
"""


def _resolve_action(
    requirement_id: str,
    action: RecoveryAction,
    state: AgentState,
) -> tuple[dict | None, dict | None]:
    """Resolve one schema-valid action against the observed state."""
    selector = action.model_dump()
    raw = {"requirement_id": requirement_id, "action": dict(selector)}
    clean = {
        "requirement_id": requirement_id,
        "action": action.tool,
        "query": action.query,
    }
    board = state["evidence_board"][requirement_id]

    if clean["action"] == "search_within_paper":
        candidates = {
            f"P{index}": item["paper"]
            for index, item in enumerate(board.get("candidate_papers", []), start=1)
        }
        candidate_id = selector["candidate_id"]
        if candidate_id not in candidates:
            return None, _rejection(raw, "unknown_candidate")
        clean.update(candidate_id=candidate_id, paper=candidates[candidate_id])
    elif clean["action"] == "expand_neighbors":
        allowed_ids = {
            item["chunk_id"] for item in state["evidence"] if item["id"] in board["evidence_ids"]
        }
        if selector["chunk_id"] not in allowed_ids:
            return None, _rejection(raw, "unlinked_chunk")
        clean["chunk_id"] = selector["chunk_id"]
    else:
        requirement = next(
            item for item in state["plan"]["requirements"] if item["id"] == requirement_id
        )
        if requirement["top_k"] >= MAX_TOP_K:
            return None, _rejection(raw, "depth_at_max")
    return clean, None


def sanitize_assessments(
    payload: object,
    state: AgentState,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Retain per-requirement coverage decisions and extract valid recovery actions."""
    try:
        raw_assessments = ControllerPayload.model_validate(payload).assessments
    except ValidationError:
        return [], [], [_rejection(payload, "invalid_assessments_payload")]

    requirement_order = [item["id"] for item in state["plan"]["requirements"]]
    requirement_ids = set(requirement_order)
    assessments = []
    actions = []
    rejections = []
    seen = set()
    for raw in raw_assessments:
        try:
            parsed = RequirementAssessment.model_validate(raw)
        except ValidationError:
            rejections.append(_rejection(raw, "invalid_assessment"))
            continue
        requirement_id = parsed.requirement_id
        if requirement_id not in requirement_ids:
            rejections.append(_rejection(raw, "unknown_requirement"))
            continue
        if requirement_id in seen:
            rejections.append(_rejection(raw, "duplicate_assessment"))
            continue
        assessment = parsed.model_dump(exclude={"action"})
        assessment["action"] = None
        assessments.append(assessment)
        seen.add(requirement_id)
        if not parsed.action:
            continue
        if len(actions) >= MAX_ACTIONS:
            rejections.append(_rejection(raw, "action_limit"))
            continue
        action, rejection = _resolve_action(requirement_id, parsed.action, state)
        if rejection:
            rejections.append(rejection)
        else:
            assessment["action"] = action
            actions.append(action)

    rejections.extend(
        _rejection({"requirement_id": requirement_id}, "missing_assessment")
        for requirement_id in requirement_order
        if requirement_id not in seen
    )
    return assessments, actions, rejections


def controller_node(state: AgentState, llm: LLMClient) -> dict:
    """Choose zero to two valid actions from one retrieval observation."""
    try:
        payload = llm.complete_json(_controller_prompt(state))
    except ValueError:
        LOGGER.warning("[controller] invalid JSON; continuing without recovery")
        assessments = []
        actions = []
        rejections = [{"requirement_id": None, "action": None, "reason": "invalid_json"}]
    else:
        assessments, actions, rejections = sanitize_assessments(payload, state)
    LOGGER.info("[controller] actions=%d rejected=%d", len(actions), len(rejections))
    return {
        "controller_trace": {
            "assessments": assessments,
            "actions": actions,
            "rejected_actions": len(rejections),
            "rejections": rejections,
        },
    }

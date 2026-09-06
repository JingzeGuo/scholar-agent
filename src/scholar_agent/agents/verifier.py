"""LLM-based evidence-coverage node."""

from __future__ import annotations

import logging
import re

from scholar_agent.agents.planner import evidence_matches_target, requirement_targets
from scholar_agent.indexes import tokenize
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
EVIDENCE_ID_RE = re.compile(r"E(\d+)")


def _matches_coverage_target(
    target: str,
    named_targets: list[str],
    item: dict,
) -> bool:
    if not evidence_matches_target(target, item):
        return False

    target_length = len(tokenize(target))
    return not any(
        other != target
        and len(tokenize(other)) > target_length
        and evidence_matches_target(other, item)
        for other in named_targets
    )


def _verifier_prompt(state: AgentState) -> str:
    plan = state["plan"]
    evidence_text = "\n".join(
        f"E{index} [{item['paper']} p.{item['page']}]: {item['text']}"
        for index, item in enumerate(state["evidence"], start=1)
    )

    requirements_text = "\n".join(
        f'{item["id"]}: {item["description"]} | targets: {item["targets"]}'
        for item in plan["requirements"]
    )

    return f"""You are the Verifier in an academic research workflow.

Decide which supplied evidence directly supports each atomic requirement.

Return one JSON object:
- "covered": requirement ID -> list of supplied evidence IDs
- "corrective_queries": one object per useful corrective query, each with a missing
  "requirement_id" and a concise English "query"; otherwise an empty list

Rules:
- Use only supplied evidence IDs.
- Return at most one corrective query per missing requirement.
- Cover a requirement only when the evidence collectively supports its entire description.
- When a requirement names targets, the evidence must collectively cover every named target.
- A comparison may be supported by combining separate evidence about each target; no passage
  needs to state the comparison directly.
- Related methods cannot substitute for a named target.
- Do not mark a requirement covered merely because the evidence is topically related.
- Partial coverage is acceptable.
- Evidence absence is preferable to unsupported approval.
- Respect constraints present in the original question without inventing new ones.

Requirements:
{requirements_text}
Question: {state["question"]}

Evidence:
{evidence_text}
"""


def _sanitize_coverage(state: AgentState, value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        LOGGER.warning("[verifier] ignored non-object coverage")
        return {}

    requirements = state["plan"]["requirements"]
    requirement_keys = {item["id"].casefold(): item for item in requirements}
    named_targets = requirement_targets(requirements)
    covered: dict[str, list[str]] = {}
    for raw_requirement_id, raw_ids in value.items():
        if not isinstance(raw_requirement_id, str) or not isinstance(raw_ids, list):
            continue
        requirement = requirement_keys.get(raw_requirement_id.casefold())
        if requirement is None:
            continue

        valid_ids: list[str] = []
        for evidence_id in raw_ids:
            match = EVIDENCE_ID_RE.fullmatch(str(evidence_id))
            if match is None:
                continue
            index = int(match.group(1))
            if not 1 <= index <= len(state["evidence"]):
                continue
            item = state["evidence"][index - 1]
            if requirement["targets"]:
                matches_intended_target = any(
                    _matches_coverage_target(target, named_targets, item)
                    for target in requirement["targets"]
                )
                matches_different_target = any(
                    evidence_matches_target(target, item) for target in named_targets
                )
                if matches_different_target and not matches_intended_target:
                    continue
            valid_ids.append(f"E{index}")

        valid_ids = list(dict.fromkeys(valid_ids))
        detectable_targets = (
            [
                target
                for target in requirement["targets"]
                if any(
                    _matches_coverage_target(target, named_targets, item)
                    for item in state["evidence"]
                )
            ]
            if len(requirement["targets"]) > 1
            else []
        )
        if valid_ids and all(
            any(
                _matches_coverage_target(
                    target,
                    named_targets,
                    state["evidence"][int(evidence_id[1:]) - 1],
                )
                for evidence_id in valid_ids
            )
            for target in detectable_targets
        ):
            covered[requirement["id"]] = valid_ids
    return covered


def verifier_node(state: AgentState, llm: LLMClient) -> dict:
    """Return complete, partial, or insufficient evidence coverage."""
    payload = llm.complete_json(_verifier_prompt(state))
    covered = _sanitize_coverage(state, payload.get("covered"))
    required = [item["id"] for item in state["plan"]["requirements"]]
    missing = [requirement_id for requirement_id in required if requirement_id not in covered]
    raw_queries = payload.get("corrective_queries", []) if missing else []
    if not isinstance(raw_queries, list):
        LOGGER.warning("[verifier] ignored non-list corrective queries")
        raw_queries = []

    missing_by_key = {requirement_id.casefold(): requirement_id for requirement_id in missing}
    corrections: dict[str, str] = {}
    for item in raw_queries:
        if not isinstance(item, dict):
            continue
        requirement_id = item.get("requirement_id")
        query = item.get("query")
        if not isinstance(requirement_id, str) or not isinstance(query, str):
            continue
        canonical_id = missing_by_key.get(requirement_id.strip().casefold())
        query = query.strip()
        if canonical_id is None or not query:
            continue
        corrections[canonical_id] = query
    corrective_queries = [
        {"requirement_id": requirement_id, "query": corrections[requirement_id]}
        for requirement_id in missing
        if requirement_id in corrections
    ]

    covered_count = len(required) - len(missing)

    if not required or covered_count == 0:
        status = "insufficient"
    elif not missing:
        status = "complete"
    else:
        status = "partial"

    verification = {
        "status": status,
        "covered": covered,
        "missing": missing,
        "corrective_queries": corrective_queries,
    }

    LOGGER.info(
        "[verifier] status=%s covered=%d/%d missing=%d",
        status,
        covered_count,
        len(required),
        len(missing),
    )

    return {"verification": verification}

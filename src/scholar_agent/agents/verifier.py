"""LLM-based evidence-coverage node."""

from __future__ import annotations

import logging
import re

from scholar_agent.agents.planner import requirement_targets, target_matches
from scholar_agent.indexes import tokenize
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
EVIDENCE_ID_RE = re.compile(r"E(\d+)")


def _matches_coverage_target(
    target: str,
    named_targets: list[str],
    text: str,
) -> bool:
    if not target_matches(target, text):
        return False

    target_length = len(tokenize(target))
    return not any(
        other != target and len(tokenize(other)) > target_length and target_matches(other, text)
        for other in named_targets
    )


def _verifier_prompt(state: AgentState) -> str:
    plan = state["plan"]
    evidence_text = "\n".join(
        f"E{index}: {item['text']}" for index, item in enumerate(state["evidence"], start=1)
    )

    requirements_text = "\n".join(
        f'{item["id"]}: {item["description"]} | targets: {item["targets"]}'
        for item in plan["requirements"]
    )

    return f"""You are the Verifier in an academic research workflow.

Decide which supplied evidence directly supports each atomic requirement.

Return one JSON object:
- "covered": requirement ID -> list of supplied evidence IDs
- "corrective_query": one concise English query for the most important missing evidence,
  or an empty string when no additional retrieval is useful

Rules:
- Use only supplied evidence IDs.
- Cover a requirement only when the evidence collectively supports its entire description.
- When a requirement names targets, the evidence must collectively cover every named target.
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
        raise ValueError("covered must be an object")

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
            if requirement["targets"] and not any(
                _matches_coverage_target(
                    target,
                    named_targets,
                    item["text"],
                )
                for target in requirement["targets"]
            ):
                continue
            valid_ids.append(f"E{index}")

        valid_ids = list(dict.fromkeys(valid_ids))
        if valid_ids and all(
            any(
                _matches_coverage_target(
                    target,
                    named_targets,
                    state["evidence"][int(evidence_id[1:]) - 1]["text"],
                )
                for evidence_id in valid_ids
            )
            for target in requirement["targets"]
        ):
            covered[requirement["id"]] = valid_ids
    return covered


def verifier_node(state: AgentState, llm: LLMClient) -> dict:
    """Return complete, partial, or insufficient evidence coverage."""
    payload = llm.complete_json(_verifier_prompt(state))
    covered = _sanitize_coverage(state, payload.get("covered"))
    raw_query = payload.get("corrective_query", "")
    if not isinstance(raw_query, str):
        raise ValueError("Verifier returned an invalid corrective query")
    corrective_query = raw_query.strip()

    required = [item["id"] for item in state["plan"]["requirements"]]
    missing = [requirement_id for requirement_id in required if requirement_id not in covered]

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
        "corrective_query": corrective_query,
    }

    LOGGER.info(
        "[verifier] status=%s covered=%d/%d missing=%d",
        status,
        covered_count,
        len(required),
        len(missing),
    )

    return {"verification": verification}

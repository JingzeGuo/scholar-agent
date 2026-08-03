"""Verifier Agent node."""

from __future__ import annotations

import logging
import re

from scholar_agent.agents.planner import target_matches
from scholar_agent.indexes import tokenize
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
EVIDENCE_ID_RE = re.compile(r"E(\d+)")


def _coverage_targets(state: AgentState) -> list[str]:
    targets = list(state["plan"]["targets"])
    return targets or ["question"]


def _matches_coverage_target(
    target: str,
    named_targets: list[str],
    text: str,
) -> bool:
    if target == "question":
        return True

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

    return f"""You are the Verifier in an academic research workflow.

Decide which supplied evidence directly supports each requested target and facet.

Return one JSON object:
- "covered": target -> facet -> list of supplied evidence IDs
- "corrective_query": one concise English query for the most important missing evidence,
  or an empty string when no additional retrieval is useful

Rules:
- Use only supplied evidence IDs.
- Related methods cannot substitute for a named target.
- Do not mark a facet covered merely because the evidence is topically related.
- Partial coverage is acceptable.
- Evidence absence is preferable to unsupported approval.
- Respect constraints present in the original question without inventing new ones.

Targets: {_coverage_targets(state)}
Facets: {plan["facets"]}
Question: {state["question"]}

Evidence:
{evidence_text}
"""


def _sanitize_coverage(state: AgentState, value: object) -> dict[str, dict[str, list[str]]]:
    if not isinstance(value, dict):
        raise ValueError("covered must be an object")
    plan = state["plan"]
    targets = _coverage_targets(state)
    target_keys = {target.casefold(): target for target in targets}
    facet_keys = {facet.casefold(): facet for facet in plan["facets"]}
    covered: dict[str, dict[str, list[str]]] = {target: {} for target in targets}
    for raw_target, raw_facets in value.items():
        if not isinstance(raw_target, str) or not isinstance(raw_facets, dict):
            continue
        target = target_keys.get(raw_target.casefold())
        if target is None:
            continue
        for raw_facet, raw_ids in raw_facets.items():
            if not isinstance(raw_facet, str) or not isinstance(raw_ids, list):
                continue
            facet = facet_keys.get(raw_facet.casefold())
            if facet is None:
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
                if target != "question" and not _matches_coverage_target(
                    target,
                    plan["targets"],
                    item["text"],
                ):
                    continue
                valid_ids.append(f"E{index}")
            if valid_ids:
                covered[target][facet] = list(dict.fromkeys(valid_ids))
    return covered


def verifier_node(state: AgentState, llm: LLMClient) -> dict:
    """Return complete, partial, or insufficient evidence coverage."""
    plan = state["plan"]
    coverage_targets = _coverage_targets(state)
    payload = llm.complete_json(_verifier_prompt(state))
    covered = _sanitize_coverage(state, payload.get("covered"))
    raw_query = payload.get("corrective_query", "")
    if not isinstance(raw_query, str):
        raise ValueError("Verifier returned an invalid corrective query")
    corrective_query = raw_query.strip()

    required = [(target, facet) for target in coverage_targets for facet in plan["facets"]]

    missing = [
        f"{target}: {facet}" for target, facet in required if not covered.get(target, {}).get(facet)
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

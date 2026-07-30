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


def _facet_matches(facet: str, text: str) -> bool:
    terms = {term for term in tokenize(facet) if len(term) > 3}
    return facet.casefold() == "mechanism" or bool(terms.intersection(tokenize(text)))


def _deterministic_coverage(state: AgentState) -> dict[str, dict[str, list[str]]]:
    plan = state["plan"]
    targets = plan["targets"] or ["question"]
    covered: dict[str, dict[str, list[str]]] = {}
    for target in targets:
        covered[target] = {}
        for facet in plan["facets"]:
            ids = []
            for index, item in enumerate(state["evidence"], start=1):
                target_ok = target == "question" or target_matches(target, item["text"])
                if target_ok and _facet_matches(facet, item["text"]):
                    ids.append(f"E{index}")
            if ids:
                covered[target][facet] = ids[:2]
    return covered


def _verifier_prompt(state: AgentState) -> str:
    plan = state["plan"]
    evidence_text = "\n".join(
        f"E{index}: {item['text']}"
        for index, item in enumerate(state["evidence"], start=1)
    )
    return f"""You are the Verifier in an academic research workflow.

Map evidence to each named target and target-level facet. Related methods must
not substitute for the named target. Return one JSON object:
- "covered": target -> facet -> list of supplied IDs such as ["E1"]
- "missing": a list of uncovered "target: facet" strings
- "corrective_query": one concise English retrieval query, empty if complete

Targets: {plan["targets"] or ["question"]}
Facets: {plan["facets"]}
Question: {state["question"]}

Evidence:
{evidence_text}
"""


def _sanitize_coverage(state: AgentState, value: object) -> dict[str, dict[str, list[str]]]:
    if not isinstance(value, dict):
        raise ValueError("covered must be an object")
    plan = state["plan"]
    targets = plan["targets"] or ["question"]
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
                if target != "question" and not target_matches(target, item["text"]):
                    continue
                valid_ids.append(f"E{index}")
            if valid_ids:
                covered[target][facet] = list(dict.fromkeys(valid_ids))
    return covered


def verifier_node(state: AgentState, llm: LLMClient | None = None) -> dict:
    """Return deterministic complete, partial, or insufficient coverage."""
    plan = state["plan"]
    targets = plan["targets"] or ["question"]
    corrective_query = ""
    if llm is None:
        covered = _deterministic_coverage(state)
    else:
        try:
            payload = llm.complete_json(_verifier_prompt(state))
            covered = _sanitize_coverage(state, payload.get("covered"))
            value = payload.get("corrective_query", "")
            corrective_query = value.strip() if isinstance(value, str) else ""
        except (KeyError, TypeError, ValueError):
            covered = _deterministic_coverage(state)

    required = [(target, facet) for target in targets for facet in plan["facets"]]
    missing = [
        f"{target}: {facet}"
        for target, facet in required
        if not covered.get(target, {}).get(facet)
    ]
    covered_count = len(required) - len(missing)
    status = "complete" if not missing else "partial" if covered_count else "insufficient"
    if missing and not corrective_query:
        corrective_query = "Find direct evidence for " + "; ".join(missing)
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

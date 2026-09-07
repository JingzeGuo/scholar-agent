"""LLM-based planning node."""

from __future__ import annotations

import logging
import re

from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
GENERIC_TARGET_SUFFIXES = {"method", "methods", "approach", "approaches", "frameworks"}
MAX_REQUIREMENTS = 5
MAX_TARGETS_PER_REQUIREMENT = 3
RETRIEVAL_STRATEGIES = frozenset({"bm25", "dense", "hybrid"})
MIN_TOP_K = 4
MAX_TOP_K = 12
DEFAULT_TOP_K = 8


def _unique_strings(values: object, limit: int) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("Expected a list")
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip() not in result:
            result.append(value.strip())
    return result[:limit]


def target_matches(target: str, text: str) -> bool:
    """Match a complete target identity rather than a substring."""
    tokens = re.findall(r"[a-z0-9]+", target.casefold())
    if not tokens:
        return False

    identity = r"[\s-]+".join(map(re.escape, tokens))
    return bool(
        re.search(
            rf"(?<![a-z0-9-]){identity}(?![a-z0-9-])",
            text.casefold(),
        )
    )


def evidence_matches_target(target: str, item: dict) -> bool:
    """Match a target in either the passage text or its source filename."""
    return any(
        isinstance(value, str) and target_matches(target, value)
        for value in (item.get("text"), item.get("paper"))
    )


def _explicit_targets(values: object, question: str) -> list[str]:
    targets: list[str] = []
    for value in _unique_strings(values, MAX_TARGETS_PER_REQUIREMENT):
        aliases = re.findall(r"\(([A-Z][A-Z0-9-]{1,9})\)", value)
        explicit = (
            value
            if target_matches(value, question)
            else next(
                (alias for alias in aliases if target_matches(alias, question)),
                "",
            )
        )
        tokens = re.findall(r"[a-z0-9]+", explicit.casefold())
        if (
            explicit
            and (not tokens or tokens[-1] not in GENERIC_TARGET_SUFFIXES)
            and explicit not in targets
        ):
            targets.append(explicit)
    return targets


def requirement_targets(requirements: list[dict]) -> list[str]:
    """Return the distinct named targets required by the plan, in plan order."""
    return list(
        dict.fromkeys(target for requirement in requirements for target in requirement["targets"]),
    )


def sanitize_retrieval_strategy(value: object) -> str:
    """Return a supported strategy, conservatively falling back to hybrid."""
    if isinstance(value, str) and value.strip().casefold() in RETRIEVAL_STRATEGIES:
        return value.strip().casefold()
    return "hybrid"


def sanitize_top_k(value: object) -> int:
    """Return a bounded retrieval depth, using the current depth for malformed values."""
    if not isinstance(value, int) or isinstance(value, bool):
        return DEFAULT_TOP_K
    return min(MAX_TOP_K, max(MIN_TOP_K, value))


def _requirements(
    values: object,
    question: str,
    limit: int = MAX_REQUIREMENTS,
) -> list[dict]:
    if not isinstance(values, list):
        raise ValueError("Expected a list")

    requirements: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        description = value.get("description")
        query = value.get("query")
        raw_targets = value.get("targets")
        if not isinstance(description, str) or not description.strip():
            continue
        if not isinstance(raw_targets, list):
            continue
        if any(not isinstance(target, str) or not target.strip() for target in raw_targets):
            continue

        supplied_targets = _unique_strings(raw_targets, MAX_TARGETS_PER_REQUIREMENT)
        targets = _explicit_targets(raw_targets, question)
        if len(targets) != len(supplied_targets):
            continue

        description = description.strip()
        identity = (description.casefold(), tuple(target.casefold() for target in targets))
        if identity in seen:
            continue
        seen.add(identity)
        requirements.append(
            {
                "id": f"R{len(requirements) + 1}",
                "description": description,
                "targets": targets,
                "query": query.strip()
                if isinstance(query, str) and query.strip()
                else description,
                "retrieval_strategy": sanitize_retrieval_strategy(
                    value.get("retrieval_strategy"),
                ),
                "top_k": sanitize_top_k(value.get("top_k")),
            },
        )
        if len(requirements) >= limit:
            break
    return requirements


def _planner_prompt(question: str) -> str:
    return f"""You plan retrieval for an evidence-grounded academic question-answering
workflow. Transform the user's question into a compact retrieval plan; do not answer the
question.

The plan is consumed as follows:
- Every requirement is retrieved independently using the strategy and depth you choose.
- All retrieved candidates are cross-encoder reranked after BM25, dense, or hybrid retrieval.
- Requirement queries and targets balance evidence selection and prevent method or aspect
  substitution.

Return one JSON object with exactly one field:
- "requirements": one to {MAX_REQUIREMENTS} objects, each with exactly these fields:
  - "description": one concise English statement of an atomic answer requirement
  - "targets": zero to {MAX_TARGETS_PER_REQUIREMENT} method or paper names explicitly written in
    the question that this requirement concerns
  - "query": one concise English evidence-seeking search query for this requirement
  - "retrieval_strategy": exactly one of "bm25", "dense", or "hybrid"
  - "top_k": an integer from {MIN_TOP_K} to {MAX_TOP_K}

Rules:
- Keep asymmetric requests separate instead of applying every aspect to every target.
- A comparison requirement may name multiple targets; a global requirement may have no targets.
- A comparison can be synthesized from separately supported facts about its targets. When those
  facts are already requirements, do not add another requirement demanding a source that directly
  compares them.
- Copy each target exactly as written in the question; do not expand or rename acronyms.
- Do not invent targets or requirements that are absent from the question.
- Preserve names and temporal constraints from the original question.
- Open-ended discovery requirements may have an empty targets list.
- Each query must target its own requirement rather than state a conclusion or answer the question.
- Choose the strategy from the nature of the requirement rather than always choosing hybrid.
- Exact paper titles, acronyms, and exact method names may favor BM25.
- Conceptual mechanisms and semantic descriptions may favor dense retrieval.
- Ambiguous comparisons or mixed lexical-semantic needs may favor hybrid retrieval.
- Broad exploratory requirements may use a larger top_k.
- Choose retrieval that efficiently finds the evidence; do not predict the answer.
- Keep the plan compact and directly grounded in the question.

User question:
<user_question>
{question}
</user_question>
"""


def planner_node(state: AgentState, llm: LLMClient) -> dict:
    """Return one compact retrieval and answer plan."""
    question = state["question"].strip()
    try:
        payload = llm.complete_json(_planner_prompt(question))
    except ValueError as exc:
        LOGGER.warning("[planner] invalid JSON; using the original question: %s", exc)
        payload = {}
    raw_requirements = payload.get("requirements") if isinstance(payload, dict) else None
    requirements = (
        _requirements(raw_requirements, question)
        if isinstance(raw_requirements, list)
        else []
    )
    if not requirements:
        LOGGER.warning("[planner] no valid requirements; using the original question")
        requirements = [
            {
                "id": "R1",
                "description": question,
                "targets": [],
                "query": question,
                "retrieval_strategy": "hybrid",
                "top_k": DEFAULT_TOP_K,
            },
        ]

    plan = {
        "requirements": requirements,
    }
    LOGGER.info(
        "[planner] requirements=%d targets=%d strategies=%s",
        len(plan["requirements"]),
        len(requirement_targets(plan["requirements"])),
        ",".join(item["retrieval_strategy"] for item in plan["requirements"]),
    )
    return {"plan": plan}

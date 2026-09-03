"""LLM-based planning node."""

from __future__ import annotations

import logging
import re

from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
GENERIC_TARGET_SUFFIXES = {"method", "methods", "approach", "approaches", "frameworks"}


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


def _explicit_targets(values: object, question: str) -> list[str]:
    targets: list[str] = []
    for value in _unique_strings(values, 3):
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


def _requirements(values: object, question: str, limit: int = 5) -> list[dict]:
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
        if not isinstance(query, str) or not query.strip():
            continue
        if not isinstance(raw_targets, list):
            continue

        supplied_targets = _unique_strings(raw_targets, 3)
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
                "query": query.strip(),
            },
        )
        if len(requirements) >= limit:
            break
    return requirements


def _planner_prompt(question: str) -> str:
    return f"""You plan retrieval and verification for an evidence-grounded academic
question-answering workflow. Transform the user's question into a compact retrieval plan;
do not answer the question.

The plan is consumed as follows:
- Every requirement query is run through both BM25 and dense retrieval.
- Every "requirement" is one independent evidence-coverage check for the Verifier.
- Requirement queries and targets are used to balance evidence selection and prevent method
  or aspect substitution.

Return one JSON object with exactly one field:
- "requirements": one to five objects, each with exactly these fields:
  - "description": one concise English statement of an independently verifiable answer requirement
  - "targets": zero to three method or paper names explicitly written in the question that this
    requirement concerns
  - "query": one concise English evidence-seeking search query for this requirement

Rules:
- Keep asymmetric requests separate instead of applying every aspect to every target.
- A comparison requirement may name multiple targets; a global requirement may have no targets.
- Do not invent targets or requirements that are absent from the question.
- Preserve names and temporal constraints from the original question.
- Open-ended discovery requirements may have an empty targets list.
- Each query must target its own requirement rather than state a conclusion or answer the question.
- Keep the plan compact and directly grounded in the question.

User question:
<user_question>
{question}
</user_question>
"""


def planner_node(state: AgentState, llm: LLMClient) -> dict:
    """Return one compact retrieval and answer plan."""
    question = state["question"].strip()
    payload = llm.complete_json(_planner_prompt(question))
    requirements = _requirements(payload.get("requirements"), question)
    if not requirements:
        raise ValueError("Planner returned no valid requirements")
    queries = _unique_strings([requirement["query"] for requirement in requirements], 5)

    plan = {
        "queries": queries,
        "requirements": requirements,
    }
    LOGGER.info(
        "[planner] queries=%d requirements=%d targets=%d",
        len(plan["queries"]),
        len(plan["requirements"]),
        len(requirement_targets(plan["requirements"])),
    )
    return {"plan": plan}

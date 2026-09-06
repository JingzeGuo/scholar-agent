"""LLM-based planning node."""

from __future__ import annotations

import logging
import re

from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
GENERIC_TARGET_SUFFIXES = {"method", "methods", "approach", "approaches", "frameworks"}
INITIALISM_SUFFIXES = {
    "approach",
    "benchmark",
    "dataset",
    "framework",
    "method",
    "model",
    "paper",
    "system",
}
MAX_REQUIREMENTS = 5
MAX_TARGETS_PER_REQUIREMENT = 3


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


def _target_initialisms(value: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", value)
    token_groups = [tokens]
    while tokens and tokens[-1].casefold() in INITIALISM_SUFFIXES:
        tokens = tokens[:-1]
        token_groups.append(tokens)

    return {
        "".join(token if token.isupper() else token[0].upper() for token in group)
        for group in token_groups
        if group
    }


def _explicit_targets(
    values: object,
    question: str,
    *,
    allow_initialisms: bool = False,
) -> list[str]:
    targets: list[str] = []
    question_initialisms = (
        set(re.findall(r"(?<![A-Z0-9-])[A-Z][A-Z0-9-]{1,9}(?![A-Z0-9-])", question))
        if allow_initialisms
        else set()
    )
    for value in _unique_strings(values, MAX_TARGETS_PER_REQUIREMENT):
        aliases = re.findall(r"\(([A-Z][A-Z0-9-]{1,9})\)", value)
        if allow_initialisms:
            aliases.extend(question_initialisms & _target_initialisms(value))
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


def _requirements(
    values: object,
    question: str,
    limit: int = MAX_REQUIREMENTS,
    *,
    allow_initialisms: bool = False,
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
        if not isinstance(query, str) or not query.strip():
            continue
        if not isinstance(raw_targets, list):
            continue

        supplied_targets = _unique_strings(raw_targets, MAX_TARGETS_PER_REQUIREMENT)
        targets = _explicit_targets(
            raw_targets,
            question,
            allow_initialisms=allow_initialisms,
        )
        if len(targets) != len(supplied_targets):
            every_target_resolved = allow_initialisms and all(
                _explicit_targets(
                    [target],
                    question,
                    allow_initialisms=True,
                )
                for target in supplied_targets
            )
            if not every_target_resolved:
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
- "requirements": one to {MAX_REQUIREMENTS} objects, each with exactly these fields:
  - "description": one concise English statement of an independently verifiable answer requirement
  - "targets": zero to {MAX_TARGETS_PER_REQUIREMENT} method or paper names explicitly written in
    the question that this requirement concerns
  - "query": one concise English evidence-seeking search query for this requirement

Rules:
- Keep asymmetric requests separate instead of applying every aspect to every target.
- A comparison requirement may name multiple targets; a global requirement may have no targets.
- A comparison can be synthesized from separately supported facts about its targets. When those
  facts are already requirements, do not add another requirement demanding a source that directly
  compares them.
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
    try:
        payload = llm.complete_json(_planner_prompt(question))
    except ValueError as exc:
        LOGGER.warning("[planner] invalid JSON; using the original question: %s", exc)
        payload = {}
    raw_requirements = payload.get("requirements")
    requirements = (
        _requirements(raw_requirements, question)
        if isinstance(raw_requirements, list)
        else []
    )
    if not requirements and isinstance(raw_requirements, list):
        requirements = _requirements(
            raw_requirements,
            question,
            allow_initialisms=True,
        )
    if not requirements:
        LOGGER.warning("[planner] no valid requirements; using the original question")
        requirements = [
            {
                "id": "R1",
                "description": question,
                "targets": [],
                "query": question,
            },
        ]

    plan = {
        "requirements": requirements,
    }
    LOGGER.info(
        "[planner] requirements=%d targets=%d",
        len(plan["requirements"]),
        len(requirement_targets(plan["requirements"])),
    )
    return {"plan": plan}

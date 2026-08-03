"""Planner Agent node."""

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


def _planner_prompt(question: str) -> str:
    return f"""You plan retrieval and verification for an evidence-grounded academic
question-answering workflow. Transform the user's question into a compact retrieval plan;
do not answer the question.

The plan is consumed as follows:
- "queries" are sent to lexical and dense retrievers.
- "entities" seed one-hop academic entity-graph retrieval.
- Every "target" x "facet" pair becomes an evidence-coverage check for the Verifier.
- "output_language" controls the language used by the Writer.

Return one JSON object with exactly these fields:
- "queries": one to three concise English search queries; preserve proper names and constraints
- "entities": zero to five important method, paper, dataset, or author names for graph retrieval
- "targets": zero to three method or paper names explicitly written in the question and requiring
  separate evidence coverage
- "facets": one to five minimal aspects required to answer the question; include only aspects
  explicitly requested or directly implied by the question type
- "output_language": the language explicitly requested by the user, otherwise the language of
  the question

Rules:
- Do not invent targets that are absent from the question.
- Do not invent requirements that the user did not ask for.
- Preserve names and temporal constraints from the original question.
- Open-ended discovery questions may have an empty targets list.
- Queries must retrieve evidence rather than state conclusions or answer the question.
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
    queries = _unique_strings(payload.get("queries"), 3)
    entities = _unique_strings(payload.get("entities"), 5)
    if not queries:
        raise ValueError("Planner returned no queries")
    facets = _unique_strings(payload.get("facets"), 5)
    if not facets:
        raise ValueError("Planner returned no facets")
    language = payload.get("output_language")
    if not isinstance(language, str) or not 0 < len(language.strip()) <= 30:
        raise ValueError("Planner returned an invalid output language")

    plan = {
        "queries": queries,
        "entities": entities,
        "targets": _explicit_targets(payload.get("targets"), question),
        "facets": facets,
        "output_language": language.strip(),
    }
    LOGGER.info(
        "[planner] queries=%d targets=%d facets=%d language=%s",
        len(plan["queries"]),
        len(plan["targets"]),
        len(plan["facets"]),
        plan["output_language"],
    )
    return {"plan": plan}

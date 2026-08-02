"""Planner Agent node."""

from __future__ import annotations

import logging
import re

from scholar_agent.graph_store import extract_entities, normalize_entity
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
METHOD_RE = re.compile(
    r"(?<![\w-])(?:[A-Z][a-z]+-[A-Z][A-Z0-9-]*|[A-Z]{2,10}|"
    r"[A-Z][a-z]+[A-Z][A-Za-z0-9]*)(?![\w-])",
)
CJK_RE = re.compile(r"[\u3400-\u9fff]")
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


def _heuristic_plan(question: str) -> dict:
    methods = _explicit_targets(
        list(dict.fromkeys(METHOD_RE.findall(question))),
        question,
    )

    entities = list(
        dict.fromkeys(
            [
                *(normalize_entity(method) for method in methods),
                *extract_entities(question),
            ]
        )
    )

    return {
        "queries": [question],
        "entities": entities[:5],
        "targets": methods[:3],
        "facets": [question],
        "output_language": "Chinese" if CJK_RE.search(question) else "English",
    }


def _planner_prompt(question: str) -> str:
    return f"""You are the Planner in an academic retrieval workflow.

Return one JSON object with exactly these fields:
- "queries": one to three concise English retrieval queries
- "entities": zero to five important methods, papers, datasets, or authors
- "targets": zero to three method or paper names explicitly written in the question
- "facets": one to five aspects explicitly requested by the user
- "output_language": the language expected by the user

Rules:
- Do not invent targets that are absent from the question.
- Do not invent requirements that the user did not ask for.
- Preserve names and temporal constraints from the original question.
- Open-ended discovery questions may have an empty targets list.
- Keep the plan compact and directly grounded in the question.

Question:
{question}
"""


def planner_node(state: AgentState, llm: LLMClient | None = None) -> dict:
    """Return one compact retrieval and answer plan."""
    question = state["question"].strip()
    fallback = _heuristic_plan(question)
    if llm is None:
        plan = fallback
    else:
        try:
            payload = llm.complete_json(_planner_prompt(question))
            queries = _unique_strings(payload.get("queries"), 3)
            entities = _unique_strings(payload.get("entities"), 5)
            if not queries:
                raise ValueError("Planner returned no queries")
            language = payload.get("output_language")
            targets = _explicit_targets(payload.get("targets"), question)
            facets = _unique_strings(payload.get("facets"), 5) or [question]

            plan = {
                "queries": queries,
                "entities": entities,
                "targets": targets,
                "facets": facets,
                "output_language": (
                    language.strip()
                    if isinstance(language, str) and 0 < len(language.strip()) <= 30
                    else fallback["output_language"]
                ),
            }
        except (KeyError, TypeError, ValueError):
            plan = fallback
    LOGGER.info(
        "[planner] queries=%d targets=%d facets=%d language=%s",
        len(plan["queries"]),
        len(plan["targets"]),
        len(plan["facets"]),
        plan["output_language"],
    )
    return {"plan": plan}

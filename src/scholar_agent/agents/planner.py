"""Planner Agent node."""

from __future__ import annotations

import logging
import re

from scholar_agent.graph_store import extract_entities
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
METHOD_RE = re.compile(
    r"(?<![\w-])(?:[A-Z][a-z]+-[A-Z][A-Z0-9-]*|[A-Z]{2,10}|"
    r"[A-Z][a-z]+[A-Z][A-Za-z0-9]*)(?![\w-])",
)
CJK_RE = re.compile(r"[\u3400-\u9fff]")
GLOBAL_FACET_TERMS = ("difference", "comparison", "comparative", "advantage", "trade-off")


def _unique_strings(values: object, limit: int) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("Expected a list")
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip() not in result:
            result.append(value.strip())
    return result[:limit]


def target_matches(target: str, text: str) -> bool:
    """Match a complete method identity, never a substring or hyphen suffix."""
    tokens = re.findall(r"[a-z0-9]+", target.casefold())
    if not tokens:
        return False
    identity = r"[\s-]+".join(map(re.escape, tokens))
    return re.search(rf"(?<![a-z0-9-]){identity}(?![a-z0-9-])", text.casefold()) is not None


def _heuristic_plan(question: str) -> dict:
    methods = list(dict.fromkeys(METHOD_RE.findall(question)))
    entities = methods + [
        entity
        for entity in extract_entities(question)
        if entity not in methods
        and not any(method.casefold() in entity.casefold() for method in methods)
    ]
    queries = [question]
    queries.extend(f"{entity} academic paper evidence" for entity in methods)
    return {
        "queries": queries[:3],
        "entities": entities[:5],
        "targets": methods[:3],
        "facets": ["mechanism"],
        "output_language": "Chinese" if CJK_RE.search(question) else "English",
    }


def _target_facets(values: object) -> list[str]:
    facets = _unique_strings(values, 5)
    return [
        facet
        for facet in facets
        if not any(term in facet.casefold() for term in GLOBAL_FACET_TERMS)
    ] or ["mechanism"]


def _planner_prompt(question: str) -> str:
    return f"""You are the Planner in an academic retrieval workflow.

Return one JSON object with exactly these fields:
- "queries": one to three concise English retrieval queries
- "entities": zero to five key methods, datasets, papers, or authors
- "targets": zero to three methods or papers the answer must discuss
- "facets": one to five target-level aspects such as retrieval trigger,
  correction mechanism, or generation control
- "output_language": the language in which the user expects the answer

Do not put comparative differences, advantages, or trade-offs in "facets";
the Writer derives comparisons from target-level evidence. Keep method names
unchanged in "targets", but expand known acronyms in retrieval queries (for
example, DPR to Dense Passage Retrieval). Do not turn a general retrieval
mechanism into a retrieval trigger unless the user asks about triggering.
Do not create subquestions, dependency graphs, priorities, or budgets.

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
            plan = {
                "queries": queries,
                "entities": entities,
                "targets": _unique_strings(payload.get("targets"), 3),
                "facets": _target_facets(payload.get("facets")),
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

"""Planner Agent node."""

from __future__ import annotations

import logging
import re

from scholar_agent.graph_store import extract_entities
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
METHOD_RE = re.compile(r"\b(?:[A-Z][a-z]+-[A-Z][A-Z0-9-]*|[A-Z]{2,10})\b")


def _unique_strings(values: object, limit: int) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("Expected a list")
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip() not in result:
            result.append(value.strip())
    return result[:limit]


def _heuristic_plan(question: str) -> tuple[list[str], list[str]]:
    methods = list(dict.fromkeys(METHOD_RE.findall(question)))
    entities = methods + [
        entity for entity in extract_entities(question) if entity not in methods
    ]
    queries = [question]
    queries.extend(f"{entity} academic paper evidence" for entity in methods)
    return queries[:3], entities[:5]


def _planner_prompt(question: str) -> str:
    return f"""You are the Planner in an academic retrieval workflow.

Return one JSON object with exactly these fields:
- "queries": one to three concise retrieval queries
- "entities": zero to five key methods, datasets, papers, or authors

Keep the user's original terminology. Do not create subquestions, dependency
graphs, priorities, or budgets. The Researcher will execute all retrievers.

Question:
{question}
"""


def planner_node(state: AgentState, llm: LLMClient | None = None) -> dict:
    """Return at most three queries and five entities."""
    question = state["question"].strip()
    if llm is None:
        queries, entities = _heuristic_plan(question)
    else:
        try:
            payload = llm.complete_json(_planner_prompt(question))
            queries = _unique_strings(payload.get("queries"), 3)
            entities = _unique_strings(payload.get("entities"), 5)
            if not queries:
                raise ValueError("Planner returned no queries")
        except (KeyError, TypeError, ValueError):
            queries, entities = [question], []
    LOGGER.info(
        "[planner] generated %d queries and %d entities",
        len(queries),
        len(entities),
    )
    return {"queries": queries, "entities": entities}

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
PRE_GENERATION_RE = re.compile(r"\b(?:before generation|pre-generation|prior to generation)\b")
OPEN_METHOD_RE = re.compile(r"\bwhich\b[^?.]*\bmethods\b", re.IGNORECASE)
GLOBAL_FACET_TERMS = ("difference", "comparison", "comparative", "advantage", "trade-off")
GENERIC_TARGETS = {"rag", "retrieval augmented generation", "llm", "lm", "qa", "nlp"}
GENERIC_TARGET_SUFFIXES = {"method", "methods", "approach", "approaches", "frameworks"}
TARGET_ALIASES = {"standard rag": ("standard rag", "vector rag", "vectorrag")}
UNASKED = ("evaluation result", "effectiveness")


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
    forms = TARGET_ALIASES.get(" ".join(tokens), (target,))
    for form in forms:
        identity = r"[\s-]+".join(
            map(re.escape, re.findall(r"[a-z0-9]+", form.casefold())),
        )
        if re.search(rf"(?<![a-z0-9-]){identity}(?![a-z0-9-])", text.casefold()):
            return True
    return False


def _explicit_targets(values: object, question: str) -> list[str]:
    targets: list[str] = []
    for value in _unique_strings(values, 3):
        aliases = re.findall(r"\(([A-Z][A-Z0-9-]{1,9})\)", value)
        explicit = value if target_matches(value, question) else next(
            (alias for alias in aliases if target_matches(alias, question)),
            "",
        )
        tokens = re.findall(r"[a-z0-9]+", explicit.casefold())
        normalized = " ".join(tokens)
        if (
            explicit
            and normalized not in GENERIC_TARGETS
            and (not tokens or tokens[-1] not in GENERIC_TARGET_SUFFIXES)
            and explicit not in targets
        ):
            targets.append(explicit)
    return targets


def _heuristic_plan(question: str) -> dict:
    methods = _explicit_targets(list(dict.fromkeys(METHOD_RE.findall(question))), question)
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
        "facets": _target_facets(["mechanism"], question),
        "output_language": "Chinese" if CJK_RE.search(question) else "English",
    }


def _facet_requested(facet: str, question: str) -> bool:
    lowered = facet.casefold()
    asked = question.casefold()
    return all(term not in lowered or term in asked for term in UNASKED)


def _target_facets(values: object, question: str) -> list[str]:
    facets = [
        facet
        for facet in _unique_strings(values, 5)
        if _facet_requested(facet, question)
        if not any(term in facet.casefold() for term in GLOBAL_FACET_TERMS)
    ] or ["mechanism"]
    if PRE_GENERATION_RE.search(question.casefold()) and not any(
        "before generation" in facet.casefold() or "pre-generation" in facet.casefold()
        for facet in facets
    ):
        facets = ["timing before generation", *facets][:5]
    has_examples = any(facet.casefold() == "method examples" for facet in facets)
    if OPEN_METHOD_RE.search(question) and not has_examples:
        facets = [facets[0], "method examples", *facets[1:]][:5]
    return facets


def _focus_open_queries(
    queries: list[str],
    entities: list[str], targets: list[str],
    question: str,
) -> list[str]:
    if not targets and PRE_GENERATION_RE.search(question.casefold()):
        return [
            "runtime retrieval evaluator relevance score corrective actions before generator",
            "retrieved passage relevance reflection tokens before generation",
            "confidence threshold trigger additional retrieval before generation",
        ]
    if targets or not entities:
        return queries
    return [f"{entity} {queries[0]}" for entity in entities[:3]]


def _planner_prompt(question: str) -> str:
    return f"""You are the Planner in an academic retrieval workflow.

Return one JSON object with exactly these fields:
- "queries": one to three concise English retrieval queries
- "entities": zero to five key methods, datasets, papers, or authors
- "targets": zero to three method or paper names written explicitly in the question
- "facets": one to five target-level aspects such as retrieval trigger,
  correction mechanism, or generation control
- "output_language": the language in which the user expects the answer

Never infer targets or choose examples: "Which methods" and "summarize three
approaches" have no targets, while "use RAPTOR and another method" has only
RAPTOR. Generic categories such as RAG are not targets. Every facet must apply
naturally to every explicit target. For method-versus-benchmark identity
questions, use symmetric facets such as identity and purpose. With no targets,
facets are question-level requirements such as evaluation dimensions and
framework examples. Derive facets from aspects actually asked; never default
to the sample facets. Hierarchical multi-section questions use facets such as
index structure, cross-section synthesis, and method examples. Preserve timing:
an in-pipeline retrieval-quality check before generation is not a post-hoc RAG
evaluation framework. Temporal qualifiers must appear in facets, not only
queries. For open discovery, entities and queries may propose likely corpus
methods even though targets stays empty. Make operational timing explicit in
retrieval queries and prefer runtime relevance gates over offline judges.
Do not put differences or trade-offs in facets. Keep method names unchanged in
targets, but expand acronyms in retrieval queries. Do not create subquestions,
dependency graphs, priorities, or budgets.

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
            plan = {
                "queries": _focus_open_queries(queries, entities, targets, question),
                "entities": entities,
                "targets": targets,
                "facets": _target_facets(payload.get("facets"), question),
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

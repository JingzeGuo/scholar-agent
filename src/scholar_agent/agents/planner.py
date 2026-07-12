"""Structured Planner: QueryPlan only (no free-form plan strings).

Offline-deterministic by default using the query classifier. Optional LLM
structured output can refine plans when configured (Phase 6 keeps tests offline).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scholar_agent.ids import make_sub_question_id, normalize_text
from scholar_agent.logging import get_logger
from scholar_agent.models.base import QueryType
from scholar_agent.models.planning import QueryPlan, SubQuestion, SubQuestionStatus
from scholar_agent.retrieval.router import classify_query_type

logger = get_logger(__name__)

_VS_SPLIT = re.compile(r"\s+(?:vs\.?|versus)\s+", re.I)
_AND_SPLIT = re.compile(r"\s+and\s+", re.I)


@dataclass
class Planner:
    """Produce a structured QueryPlan for the multi-agent workflow."""

    def plan(self, query: str) -> QueryPlan:
        query = query.strip()
        if not query:
            raise ValueError("query must be non-empty")

        qtype, signals = classify_query_type(query)
        plan_seed = normalize_text(query)[:64]

        if qtype == QueryType.COMPARISON:
            sub_qs = self._plan_comparison(query, plan_seed)
            answer_type = "comparison"
            diversity = 2
        elif qtype == QueryType.SYNTHESIS:
            sub_qs = self._plan_synthesis(query, plan_seed)
            answer_type = "synthesis"
            diversity = 3
        elif qtype == QueryType.RELATIONAL:
            sub_qs = [
                SubQuestion(
                    id=make_sub_question_id(plan_seed, query, 0),
                    question=query,
                    query_type=QueryType.RELATIONAL,
                    required_evidence=["relation", "entities", "supporting passage"],
                    status=SubQuestionStatus.PENDING,
                )
            ]
            answer_type = "relational"
            diversity = 2
        else:
            # Simple factual / keyword / semantic: avoid unnecessary decomposition
            sub_qs = [
                SubQuestion(
                    id=make_sub_question_id(plan_seed, query, 0),
                    question=query,
                    query_type=qtype,
                    required_evidence=["definition_or_fact", "supporting passage"],
                    status=SubQuestionStatus.PENDING,
                )
            ]
            answer_type = "factual" if qtype == QueryType.KEYWORD else "semantic"
            diversity = 1

        plan = QueryPlan(
            original_query=query,
            answer_type=answer_type,
            sub_questions=sub_qs,
            expected_source_diversity=diversity,
        )
        logger.info(
            "planned query_type=%s sub_questions=%s signals=%s",
            qtype.value,
            len(sub_qs),
            signals[:5],
        )
        return plan

    def _plan_comparison(self, query: str, plan_seed: str) -> list[SubQuestion]:
        parts = _VS_SPLIT.split(query)
        entities: list[str] = []
        if len(parts) == 2:
            # "Compare Self-RAG versus CRAG" or "Self-RAG vs CRAG"
            left = re.sub(
                r"^(compare|comparison of|differences between)\s+",
                "",
                parts[0],
                flags=re.I,
            ).strip(" ?")
            right = parts[1].strip(" ?")
            # strip trailing "in ..." clauses lightly
            right = re.split(r"\s+in\s+|\s+on\s+|\s+for\s+", right, maxsplit=1)[0].strip()
            if left and right:
                entities = [left, right]

        sub_qs: list[SubQuestion] = []
        if len(entities) == 2:
            a, b = entities[0], entities[1]
            questions = [
                (f"What is {a}?", QueryType.SEMANTIC, ["definition", a]),
                (f"What is {b}?", QueryType.SEMANTIC, ["definition", b]),
                (
                    f"How do {a} and {b} differ regarding retrieval and correction?",
                    QueryType.COMPARISON,
                    ["differences", a, b],
                ),
            ]
        else:
            questions = [
                (query, QueryType.COMPARISON, ["comparison", "both sides"]),
                (
                    f"Key differences and trade-offs for: {query}",
                    QueryType.COMPARISON,
                    ["trade-offs"],
                ),
            ]

        for i, (q, qt, req) in enumerate(questions):
            sub_qs.append(
                SubQuestion(
                    id=make_sub_question_id(plan_seed, q, i),
                    question=q,
                    query_type=qt,
                    required_evidence=req,
                    status=SubQuestionStatus.PENDING,
                )
            )
        return sub_qs

    def _plan_synthesis(self, query: str, plan_seed: str) -> list[SubQuestion]:
        questions = [
            (query, QueryType.SYNTHESIS, ["main themes", "representative methods"]),
            (
                f"What methods or systems are most central to: {query}",
                QueryType.KEYWORD,
                ["methods", "systems"],
            ),
            (
                f"What open challenges remain regarding: {query}",
                QueryType.SEMANTIC,
                ["limitations", "open problems"],
            ),
        ]
        return [
            SubQuestion(
                id=make_sub_question_id(plan_seed, q, i),
                question=q,
                query_type=qt,
                required_evidence=req,
                status=SubQuestionStatus.PENDING,
            )
            for i, (q, qt, req) in enumerate(questions)
        ]

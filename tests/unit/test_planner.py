"""Tests for structured Planner."""

from __future__ import annotations

from scholar_agent.agents.planner import Planner
from scholar_agent.models.base import QueryType


def test_simple_factual_not_over_decomposed() -> None:
    plan = Planner().plan("What is RAPTOR?")
    assert len(plan.sub_questions) == 1
    assert plan.sub_questions[0].query_type in {QueryType.SEMANTIC, QueryType.KEYWORD}
    assert plan.original_query == "What is RAPTOR?"


def test_comparison_produces_multiple_subquestions() -> None:
    plan = Planner().plan("Compare Self-RAG versus CRAG")
    assert plan.answer_type == "comparison"
    assert len(plan.sub_questions) >= 2
    assert plan.expected_source_diversity >= 2
    ids = [sq.id for sq in plan.sub_questions]
    assert len(ids) == len(set(ids))


def test_synthesis_plan() -> None:
    plan = Planner().plan("Summarize main trends across agentic RAG papers")
    assert plan.answer_type == "synthesis"
    assert len(plan.sub_questions) >= 2


def test_plan_is_structured_not_string() -> None:
    plan = Planner().plan("Which datasets does DPR evaluate on?")
    assert hasattr(plan, "sub_questions")
    assert all(hasattr(sq, "required_evidence") for sq in plan.sub_questions)

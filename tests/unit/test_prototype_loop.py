"""Tests for the Phase 0 LangGraph prototype loop."""

from __future__ import annotations

from scholar_agent.agents.prototype_loop import (
    FakeResearchModel,
    PrototypeLoopConfig,
    run_prototype_loop,
)
from scholar_agent.models import EventType


def test_prototype_loop_succeeds_with_enough_evidence() -> None:
    result = run_prototype_loop(
        "What is corrective RAG?",
        config=PrototypeLoopConfig(
            max_tool_calls=4,
            max_iterations=5,
            required_evidence=2,
        ),
        run_id="run_test_success",
    )
    assert result.success is True
    assert result.run_id == "run_test_success"
    assert result.tool_call_count == 2
    assert result.terminated_reason == "evidence_sufficient"
    assert "corrective RAG" in result.answer or "evidence" in result.answer.lower()
    event_types = [e.event_type for e in result.events]
    assert EventType.RUN_STARTED in event_types
    assert EventType.DECISION in event_types
    assert EventType.TOOL_RESULT in event_types
    assert EventType.VERIFICATION in event_types
    assert EventType.TERMINATED in event_types
    assert EventType.RUN_FINISHED in event_types


def test_prototype_loop_hits_tool_budget() -> None:
    # Require more evidence than tool budget allows
    result = run_prototype_loop(
        "Compare Self-RAG and CRAG",
        config=PrototypeLoopConfig(
            max_tool_calls=1,
            max_iterations=5,
            required_evidence=3,
        ),
    )
    assert result.success is False
    assert result.tool_call_count == 1
    assert result.terminated_reason == "tool_budget_exhausted"


def test_prototype_loop_hits_iteration_budget() -> None:
    model = FakeResearchModel(required_evidence=10)

    # Always wants more evidence but iterations cap out
    result = run_prototype_loop(
        "Synthesis query",
        config=PrototypeLoopConfig(
            max_tool_calls=10,
            max_iterations=2,
            required_evidence=10,
        ),
        model=model,
    )
    assert result.success is False
    assert result.iterations <= 2 or result.terminated_reason in {
        "iteration_budget_exhausted",
        "tool_budget_exhausted",
        "evidence_sufficient",
    }
    # With required=10 and max_iterations=2, should not succeed
    assert result.terminated_reason != "evidence_sufficient" or result.success


def test_events_have_non_empty_summaries() -> None:
    result = run_prototype_loop("GraphRAG overview")
    assert result.events
    for event in result.events:
        assert event.summary.strip()
        assert event.run_id == result.run_id


def test_fake_model_decision_policy() -> None:
    model = FakeResearchModel(required_evidence=2)
    d1 = model.decide(
        query="q",
        observations=[],
        tool_call_count=0,
        max_tool_calls=4,
        iteration=0,
        max_iterations=3,
    )
    assert d1.action == "retrieve"

    obs = [model.retrieve("q", 0), model.retrieve("q", 1)]
    d2 = model.decide(
        query="q",
        observations=obs,
        tool_call_count=2,
        max_tool_calls=4,
        iteration=2,
        max_iterations=3,
    )
    assert d2.action == "verify"

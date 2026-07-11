"""Tests for core Phase 0 models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scholar_agent.models import (
    BudgetStatus,
    EventType,
    ExecutionEvent,
    TokenUsage,
    new_run_id,
)


def test_run_id_format() -> None:
    rid = new_run_id()
    assert rid.startswith("run_")
    assert len(rid) == 4 + 16


def test_execution_event_requires_summary() -> None:
    with pytest.raises(ValidationError):
        ExecutionEvent(
            run_id="run_x",
            event_type=EventType.DECISION,
            component="test",
            summary="   ",
        )


def test_budget_status_exhaustion() -> None:
    status = BudgetStatus(tool_call_count=4, max_tool_calls=4)
    assert status.is_exhausted()
    assert status.tool_budget_remaining() == 0

    ok = BudgetStatus(tool_call_count=1, max_tool_calls=4, iteration=0, max_iterations=3)
    assert not ok.is_exhausted()


def test_token_usage_add() -> None:
    a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    b = TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    c = a.add(b)
    assert c.prompt_tokens == 13
    assert c.completion_tokens == 7
    assert c.total_tokens == 20

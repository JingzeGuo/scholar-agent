"""Shared agent state helpers for LangGraph reducers.

Phase 0 defines reducer utilities used by the prototype loop. Full Research
workflow state expands in later phases.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from scholar_agent.models import ExecutionEvent


def append_events(
    existing: list[ExecutionEvent],
    new: list[ExecutionEvent] | ExecutionEvent,
) -> list[ExecutionEvent]:
    """Append execution events without deduplicating (events are unique by id)."""
    additions = [new] if isinstance(new, ExecutionEvent) else list(new)
    return list(existing) + additions


def replace_if_set(existing: str | None, new: str | None) -> str | None:
    """LangGraph-style last-write-wins reducer for optional strings."""
    return existing if new is None else new


class BaseAgentState(TypedDict, total=False):
    """Minimal shared keys; concrete graphs extend this."""

    run_id: str
    query: str
    iteration: int
    tool_call_count: int
    events: Annotated[list[ExecutionEvent], append_events]
    terminated_reason: str | None

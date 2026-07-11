"""Shared agent state helpers for LangGraph reducers.

Reducers for execution events and evidence must append / dedupe rather than
blindly overwriting repeated retrieval results.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from scholar_agent.models import EvidenceItem, EvidenceLedger, ExecutionEvent


def append_events(
    existing: list[ExecutionEvent],
    new: list[ExecutionEvent] | ExecutionEvent,
) -> list[ExecutionEvent]:
    """Append execution events without deduplicating (events are unique by id)."""
    additions = [new] if isinstance(new, ExecutionEvent) else list(new)
    return list(existing) + additions


def merge_evidence(
    existing: list[EvidenceItem] | EvidenceLedger,
    new: list[EvidenceItem] | EvidenceItem | EvidenceLedger,
) -> list[EvidenceItem]:
    """Merge evidence with chunk+span deduplication (prefer higher scores)."""
    if isinstance(existing, EvidenceLedger):
        base = existing
    else:
        base = EvidenceLedger(items=list(existing))

    if isinstance(new, EvidenceLedger):
        incoming: list[EvidenceItem] | EvidenceItem = new.items
    else:
        incoming = new

    return base.merge(incoming).items


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
    evidence: Annotated[list[EvidenceItem], merge_evidence]
    terminated_reason: str | None

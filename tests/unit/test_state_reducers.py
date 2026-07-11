"""Tests for LangGraph-oriented state reducers (Phase 0/1)."""

from __future__ import annotations

from scholar_agent.agents.state import append_events, merge_evidence, replace_if_set
from scholar_agent.models import EventType, EvidenceItem, ExecutionEvent


def _evt(summary: str) -> ExecutionEvent:
    return ExecutionEvent(
        run_id="run_test",
        event_type=EventType.DECISION,
        component="test",
        summary=summary,
    )


def test_append_events_merges_lists() -> None:
    a = _evt("a")
    b = _evt("b")
    c = _evt("c")
    merged = append_events([a], [b, c])
    assert [e.summary for e in merged] == ["a", "b", "c"]


def test_append_events_accepts_single() -> None:
    a = _evt("a")
    b = _evt("b")
    merged = append_events([a], b)
    assert len(merged) == 2
    assert merged[1].summary == "b"


def test_replace_if_set() -> None:
    assert replace_if_set("old", None) == "old"
    assert replace_if_set("old", "new") == "new"
    assert replace_if_set(None, "new") == "new"


def test_merge_evidence_dedupes() -> None:
    a = EvidenceItem.build(
        run_id="run_1",
        sub_question_id="sq_1",
        claim="c",
        evidence_text="span",
        paper_id="p",
        chunk_id="ch",
        page_start=1,
        page_end=1,
        retrieval_method="dense",
        retrieval_score=0.1,
    )
    b = EvidenceItem.build(
        run_id="run_1",
        sub_question_id="sq_1",
        claim="c",
        evidence_text="SPAN",
        paper_id="p",
        chunk_id="ch",
        page_start=1,
        page_end=1,
        retrieval_method="sparse",
        retrieval_score=0.8,
    )
    merged = merge_evidence([a], b)
    assert len(merged) == 1
    assert merged[0].retrieval_score == 0.8

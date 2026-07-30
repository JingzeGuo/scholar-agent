from __future__ import annotations

from typing import Any

import scholar_agent.reranker
import scholar_agent.workflow as workflow_module
from scholar_agent.config import Settings
from scholar_agent.models import AgentState
from scholar_agent.workflow import initial_state, run_question


class FakeEngine:
    def __init__(self, results: list[dict], chunks: list[dict] | None = None) -> None:
        self.results = results
        self.chunks = chunks or results
        self.calls = 0

    def sparse_search(self, queries: list[str]) -> list[dict]:
        self.calls += 1
        return self.results

    def dense_search(self, queries: list[str]) -> list[dict]:
        return self.results

    def graph_search(self, entities: list[str]) -> list[dict]:
        return self.results


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool) -> list[float]:
        return [5.0] * len(pairs)


def test_langgraph_completes_four_agent_flow(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:2])
    monkeypatch.setattr(scholar_agent.reranker, "_cross_encoder", lambda model: FakeCrossEncoder())

    result = run_question(
        "Compare Self-RAG and CRAG",
        engine,  # type: ignore[arg-type]
        Settings(),
    )

    assert result["verification"]["status"] == "complete"
    assert result["retry_count"] == 0
    assert "[Self-RAG.pdf p.1]" in result["answer"]
    assert "[CRAG.pdf p.2]" in result["answer"]


def test_no_relevant_evidence_abstains_without_retry() -> None:
    engine = FakeEngine([])
    result = run_question(
        "Evidence that does not exist",
        engine,  # type: ignore[arg-type]
        Settings(),
    )

    assert result["verification"]["status"] == "insufficient"
    assert result["stop_reason"] == "no_relevant_evidence"
    assert result["retry_count"] == 0
    assert engine.calls == 1
    assert "sufficiently relevant evidence" in result["answer"]


def test_partial_workflow_retries_exactly_once(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:1], sample_chunks[:2])
    monkeypatch.setattr(scholar_agent.reranker, "_cross_encoder", lambda model: FakeCrossEncoder())

    result = run_question(
        "Compare Self-RAG and CRAG",
        engine,  # type: ignore[arg-type]
        Settings(),
    )

    assert result["verification"]["status"] == "partial"
    assert result["retry_count"] == 1
    assert engine.calls == 2
    assert "Missing evidence" in result["answer"]


def test_run_question_sets_recursion_limit(monkeypatch: Any) -> None:
    class CapturingWorkflow:
        config: dict | None = None

        def invoke(self, state: AgentState, config: dict) -> AgentState:
            self.config = config
            return state

    compiled = CapturingWorkflow()
    monkeypatch.setattr(workflow_module, "build_workflow", lambda *args: compiled)

    result = run_question("question", FakeEngine([]), Settings())  # type: ignore[arg-type]

    assert result == initial_state("question")
    assert compiled.config == {"recursion_limit": 10}


def test_agent_state_has_seven_cross_agent_fields() -> None:
    assert set(AgentState.__annotations__) == {
        "question",
        "plan",
        "evidence",
        "verification",
        "retry_count",
        "stop_reason",
        "answer",
    }

from __future__ import annotations

from typing import Any

import scholar_agent.reranker
import scholar_agent.workflow as workflow_module
from scholar_agent.config import Settings
from scholar_agent.models import AgentState
from scholar_agent.workflow import initial_state, run_question


class FakeEngine:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
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
        return [float(len(pairs) - index) for index in range(len(pairs))]


class EmptyFeedbackLLM:
    def complete_json(self, prompt: str) -> dict:
        if "Planner" in prompt:
            return {
                "queries": ["Compare Self-RAG and CRAG"],
                "entities": ["Self-RAG", "CRAG"],
            }
        return {"sufficient": False, "feedback": ""}

    def complete(self, prompt: str) -> str:
        return "The retrieved evidence remains incomplete. [E1]"


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

    assert result["sufficient"] is True
    assert result["retry_count"] == 0
    assert "candidates" not in result
    assert "[Self-RAG.pdf p.1]" in result["answer"]
    assert "[CRAG.pdf p.2]" in result["answer"]


def test_insufficient_workflow_retries_exactly_once(monkeypatch: Any) -> None:
    engine = FakeEngine([])
    result = run_question(
        "Evidence that does not exist",
        engine,  # type: ignore[arg-type]
        Settings(),
    )

    assert result["sufficient"] is False
    assert result["retry_count"] == 1
    assert engine.calls == 2
    assert "insufficient" in result["answer"]


def test_empty_llm_feedback_cannot_loop_forever(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:2])
    monkeypatch.setattr(scholar_agent.reranker, "_cross_encoder", lambda model: FakeCrossEncoder())

    result = run_question(
        "Compare Self-RAG and CRAG",
        engine,  # type: ignore[arg-type]
        Settings(),
        EmptyFeedbackLLM(),  # type: ignore[arg-type]
    )

    assert result["sufficient"] is False
    assert result["retry_count"] == 1
    assert result["feedback"]
    assert engine.calls == 2


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


def test_agent_state_has_only_cross_agent_fields() -> None:
    assert "candidates" not in AgentState.__annotations__
    assert len(AgentState.__annotations__) == 8

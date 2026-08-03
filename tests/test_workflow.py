from __future__ import annotations

from typing import Any

import pytest

import scholar_agent.reranker
import scholar_agent.workflow as workflow_module
from scholar_agent.config import Settings
from scholar_agent.models import AgentState
from scholar_agent.workflow import initial_state, route_after_verification, run_question


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


class FakeLLM:
    def complete_json(self, prompt: str) -> dict:
        if "You are the Planner" in prompt:
            return {
                "queries": ["Self-RAG CRAG retrieval"],
                "entities": ["Self-RAG", "CRAG"],
                "targets": ["Self-RAG", "CRAG"],
                "facets": ["retrieval"],
                "output_language": "English",
            }
        if "E2:" in prompt:
            covered = {
                "Self-RAG": {"retrieval": ["E1"]},
                "CRAG": {"retrieval": ["E2"]},
            }
        else:
            covered = {"Self-RAG": {"retrieval": ["E1"]}, "CRAG": {}}
        return {"covered": covered, "corrective_query": "Find CRAG retrieval evidence"}

    def complete(self, prompt: str) -> str:
        if "Status: complete" in prompt:
            return "Self-RAG uses adaptive retrieval [E1]. CRAG uses corrective retrieval [E2]."
        return "Self-RAG uses adaptive retrieval [E1]."


def _retrieval_plan(state: AgentState, llm: object) -> dict:
    return {
        "plan": {
            "queries": [state["question"]],
            "entities": ["Self-RAG", "CRAG"],
            "targets": ["Self-RAG", "CRAG"],
            "facets": ["retrieval"],
            "output_language": "English",
        },
    }


def test_langgraph_completes_four_agent_flow(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:2])
    monkeypatch.setattr(scholar_agent.reranker, "_cross_encoder", lambda model: FakeCrossEncoder())
    monkeypatch.setattr(workflow_module, "planner_node", _retrieval_plan)

    result = run_question(
        "Compare Self-RAG and CRAG",
        engine,  # type: ignore[arg-type]
        Settings(),
        FakeLLM(),  # type: ignore[arg-type]
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
        FakeLLM(),  # type: ignore[arg-type]
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
    monkeypatch.setattr(workflow_module, "planner_node", _retrieval_plan)
    researcher_calls = 0
    verifier_calls = 0
    original_researcher = workflow_module.researcher_node
    original_verifier = workflow_module.verifier_node

    def counting_researcher(*args: Any, **kwargs: Any) -> dict:
        nonlocal researcher_calls
        researcher_calls += 1
        return original_researcher(*args, **kwargs)

    def counting_verifier(*args: Any, **kwargs: Any) -> dict:
        nonlocal verifier_calls
        verifier_calls += 1
        return original_verifier(*args, **kwargs)

    monkeypatch.setattr(workflow_module, "researcher_node", counting_researcher)
    monkeypatch.setattr(workflow_module, "verifier_node", counting_verifier)

    result = run_question(
        "Compare Self-RAG and CRAG",
        engine,  # type: ignore[arg-type]
        Settings(),
        FakeLLM(),  # type: ignore[arg-type]
    )

    assert result["verification"]["status"] == "partial"
    assert result["retry_count"] == 1
    assert result["stop_reason"] == "no_new_evidence"
    assert researcher_calls == 2
    assert verifier_calls == 1
    assert "Missing evidence" in result["answer"]


def test_run_question_starts_with_initial_state(monkeypatch: Any) -> None:
    class CapturingWorkflow:
        state: AgentState | None = None

        def invoke(self, state: AgentState) -> AgentState:
            self.state = state
            return state

    compiled = CapturingWorkflow()
    monkeypatch.setattr(workflow_module, "build_workflow", lambda *args: compiled)

    result = run_question(
        "question",
        FakeEngine([]),  # type: ignore[arg-type]
        Settings(),
        FakeLLM(),  # type: ignore[arg-type]
    )

    assert result == initial_state("question")
    assert compiled.state == initial_state("question")


def test_initial_state_does_not_invent_a_facet() -> None:
    state = initial_state("Compare two methods")

    assert state["plan"]["facets"] == []


def test_workflow_requires_an_llm() -> None:
    with pytest.raises(ValueError, match="llm is required"):
        workflow_module.build_workflow(FakeEngine([]), Settings(), None)  # type: ignore[arg-type]


def test_verification_retry_limit_is_configurable() -> None:
    state = initial_state("question")
    state["verification"]["corrective_query"] = "Find missing evidence"
    state["retry_count"] = 1

    assert route_after_verification(state, Settings(max_retries=2)) == "researcher"

    state["retry_count"] = 2
    assert route_after_verification(state, Settings(max_retries=2)) == "writer"

    state["retry_count"] = 0
    state["verification"]["corrective_query"] = ""
    assert route_after_verification(state, Settings(max_retries=2)) == "writer"


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

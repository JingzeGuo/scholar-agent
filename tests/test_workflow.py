from __future__ import annotations

from typing import Any

import pytest

import scholar_agent.reranker
import scholar_agent.workflow as workflow_module
from scholar_agent.agents.writer import SAFE_ABSTENTION
from scholar_agent.config import Settings
from scholar_agent.models import AgentState
from scholar_agent.workflow import build_workflow, initial_state, run_question


class FakeEngine:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.chunks = results
        self.sparse_calls: list[tuple[list[str], int]] = []
        self.dense_calls: list[tuple[list[str], int]] = []

    def sparse_search(self, queries: list[str], top_k: int = 8) -> list[dict]:
        self.sparse_calls.append((queries, top_k))
        return self.results

    def dense_search_many(
        self,
        queries: list[str],
        top_k: int = 8,
    ) -> list[list[dict]]:
        self.dense_calls.append((queries, top_k))
        return [self.results for _ in queries]


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool) -> list[float]:
        return [5.0] * len(pairs)


class FakeLLM:
    def __init__(self) -> None:
        self.json_calls = 0

    def complete_json(self, prompt: str) -> dict:
        self.json_calls += 1
        return {
            "requirements": [
                {
                    "description": "Answer the requested evidence question",
                    "targets": [],
                    "query": "Self-RAG CRAG retrieval",
                    "retrieval_strategy": "bm25",
                    "top_k": 6,
                },
            ],
        }

    def complete(self, prompt: str) -> str:
        return "Self-RAG uses adaptive retrieval [E1]. CRAG uses corrective retrieval [E2]."


def test_adaptive_workflow_reaches_writer_and_validates_citations(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:2])
    monkeypatch.setattr(
        scholar_agent.reranker,
        "_cross_encoder",
        lambda model: FakeCrossEncoder(),
    )

    result = run_question(
        "Compare Self-RAG and CRAG",
        engine,  # type: ignore[arg-type]
        Settings(),
        FakeLLM(),  # type: ignore[arg-type]
    )

    assert result["retrieval_mode"] == "adaptive"
    assert engine.sparse_calls == [(["Self-RAG CRAG retrieval"], 6)]
    assert engine.dense_calls == []
    assert "[Self-RAG.pdf p.1]" in result["answer"]
    assert "[CRAG.pdf p.2]" in result["answer"]
    assert result["retrieval_trace"] == [
        {
            "requirement_id": "R1",
            "query": "Self-RAG CRAG retrieval",
            "retrieval_strategy": "bm25",
            "top_k": 6,
        },
    ]


def test_fixed_hybrid_workflow_ignores_planner_strategy(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:2])
    monkeypatch.setattr(
        scholar_agent.reranker,
        "_cross_encoder",
        lambda model: FakeCrossEncoder(),
    )

    result = run_question(
        "Compare Self-RAG and CRAG",
        engine,  # type: ignore[arg-type]
        Settings(),
        FakeLLM(),  # type: ignore[arg-type]
        retrieval_mode="fixed_hybrid",
    )

    assert engine.sparse_calls == [(["Self-RAG CRAG retrieval"], 6)]
    assert engine.dense_calls == [(["Self-RAG CRAG retrieval"], 6)]
    assert result["retrieval_trace"][0]["retrieval_strategy"] == "hybrid"


def test_shared_plan_skips_planner_and_is_not_mutated(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    engine = FakeEngine(sample_chunks[:1])
    monkeypatch.setattr(
        scholar_agent.reranker,
        "_cross_encoder",
        lambda model: FakeCrossEncoder(),
    )
    shared_plan = {
        "requirements": [
            {
                "id": "R1",
                "description": "Explain Self-RAG",
                "targets": ["Self-RAG"],
                "query": "Self-RAG retrieval",
                "retrieval_strategy": "bm25",
                "top_k": 6,
            },
        ],
    }
    llm = FakeLLM()

    result = run_question(
        "Explain Self-RAG",
        engine,  # type: ignore[arg-type]
        Settings(),
        llm,  # type: ignore[arg-type]
        shared_plan=shared_plan,
    )

    assert llm.json_calls == 0
    assert result["plan"] == shared_plan
    assert result["plan"] is not shared_plan
    assert shared_plan["requirements"][0]["retrieval_strategy"] == "bm25"


def test_empty_evidence_produces_deterministic_abstention() -> None:
    engine = FakeEngine([])

    result = run_question(
        "Evidence that does not exist",
        engine,  # type: ignore[arg-type]
        Settings(),
        FakeLLM(),  # type: ignore[arg-type]
    )

    assert result["evidence"] == []
    assert result["answer"] == SAFE_ABSTENTION


def test_run_question_starts_with_the_selected_initial_state(monkeypatch: Any) -> None:
    captured: list[dict] = []

    class FakeWorkflow:
        def invoke(self, state: dict) -> dict:
            captured.append(state)
            return state

    monkeypatch.setattr(
        workflow_module,
        "build_workflow",
        lambda *args, **kwargs: FakeWorkflow(),
    )

    result = run_question(
        "question",
        FakeEngine([]),  # type: ignore[arg-type]
        Settings(retrieval_mode="fixed_hybrid"),
        FakeLLM(),  # type: ignore[arg-type]
    )

    assert captured == [initial_state("question", "fixed_hybrid")]
    assert result["retrieval_mode"] == "fixed_hybrid"


def test_initial_state_is_minimal_and_does_not_invent_requirements() -> None:
    state = initial_state("question")

    assert state == {
        "question": "question",
        "retrieval_mode": "adaptive",
        "plan": {"requirements": []},
        "evidence": [],
        "retrieval_trace": [],
        "answer": "",
    }


def test_workflow_requires_an_llm() -> None:
    with pytest.raises(ValueError, match="llm is required"):
        build_workflow(FakeEngine([]), Settings(), None)  # type: ignore[arg-type]


@pytest.mark.parametrize("entrypoint", ["build", "initial"])
def test_workflow_rejects_unknown_retrieval_mode(entrypoint: str) -> None:
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        if entrypoint == "build":
            build_workflow(  # type: ignore[arg-type]
                FakeEngine([]),
                Settings(),
                FakeLLM(),  # type: ignore[arg-type]
                retrieval_mode="automatic",
            )
        else:
            initial_state("question", "automatic")


def test_agent_state_contains_only_live_workflow_fields() -> None:
    assert set(AgentState.__annotations__) == {
        "question",
        "retrieval_mode",
        "plan",
        "evidence",
        "retrieval_trace",
        "answer",
    }

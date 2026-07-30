from __future__ import annotations

from typing import Any

import scholar_agent.reranker
from scholar_agent.config import Settings
from scholar_agent.workflow import run_question


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

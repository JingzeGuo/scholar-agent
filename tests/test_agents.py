from __future__ import annotations

from typing import Any

from scholar_agent.agents.planner import planner_node, target_matches
from scholar_agent.agents.researcher import researcher_node
from scholar_agent.agents.verifier import verifier_node
from scholar_agent.agents.writer import writer_node
from scholar_agent.config import Settings
from scholar_agent.workflow import initial_state


class StubLLM:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def complete_json(self, prompt: str) -> dict[str, Any]:
        if isinstance(self.payload, Exception):
            raise self.payload
        assert "Question" in prompt
        return self.payload  # type: ignore[return-value]


class FakeEngine:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks

    def sparse_search(self, queries: list[str]) -> list[dict]:
        return self.chunks

    def dense_search(self, queries: list[str]) -> list[dict]:
        return self.chunks

    def graph_search(self, entities: list[str]) -> list[dict]:
        return self.chunks


def test_planner_returns_compact_bounded_plan() -> None:
    payload = {
        "queries": ["q1", "q2", "q3", "q4"],
        "entities": ["a", "b", "c", "d", "e", "f"],
        "targets": ["a", "b", "c", "d"],
        "facets": ["retrieval trigger", "key differences", "generation control"],
        "output_language": "Chinese",
    }
    plan = planner_node(initial_state("Compare systems"), StubLLM(payload))["plan"]  # type: ignore[arg-type]

    assert plan["queries"] == ["q1", "q2", "q3"]
    assert plan["entities"] == ["a", "b", "c", "d", "e"]
    assert plan["targets"] == ["a", "b", "c"]
    assert plan["facets"] == ["retrieval trigger", "generation control"]
    assert plan["output_language"] == "Chinese"


def test_planner_fallback_detects_methods_and_chinese() -> None:
    plan = planner_node(
        initial_state("用中文比较 DPR 和 ColBERT"),
        StubLLM(ValueError("bad JSON")),  # type: ignore[arg-type]
    )["plan"]

    assert plan["targets"] == ["DPR", "ColBERT"]
    assert plan["queries"][1:] == [
        "DPR academic paper evidence",
        "ColBERT academic paper evidence",
    ]
    assert plan["output_language"] == "Chinese"


def test_target_matching_preserves_method_identity() -> None:
    assert target_matches("Self-RAG", "Self RAG uses reflection tokens.")
    assert target_matches("CRAG", "CRAG uses a retrieval evaluator.")
    assert not target_matches("CRAG", "Self-CRAG combines both methods.")
    assert not target_matches("DPR", "ANCE uses one dense embedding.")
    assert not target_matches("RAG", "CRAG, Self-RAG, and GraphRAG are methods.")


def test_researcher_rejects_every_below_threshold_chunk(sample_chunks: list[dict]) -> None:
    state = initial_state("Compare Self-RAG and CRAG")
    state["plan"] = {
        "queries": ["Self-RAG CRAG"],
        "entities": ["Self-RAG", "CRAG"],
        "targets": ["Self-RAG", "CRAG"],
        "facets": ["mechanism"],
        "output_language": "English",
    }

    def low_scores(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        return [{**item, "score": 0.4} for item in candidates]

    result = researcher_node(
        state,
        FakeEngine(sample_chunks),  # type: ignore[arg-type]
        Settings(min_rerank_score=0.5),
        low_scores,
    )

    assert result["evidence"] == []
    assert result["stop_reason"] == "no_relevant_evidence"

    state["plan"]["targets"] = ["RoseTTAFold All-Atom"]

    def high_scores(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        return [{**item, "score": 9.0} for item in candidates]

    absent_target = researcher_node(
        state,
        FakeEngine(sample_chunks),  # type: ignore[arg-type]
        Settings(),
        high_scores,
    )
    assert absent_target["evidence"] == []
    assert absent_target["stop_reason"] == "no_relevant_evidence"


def test_researcher_merges_retry_and_balances_targets(sample_chunks: list[dict]) -> None:
    old = [
        {
            **sample_chunks[0],
            "chunk_id": f"self-{index}",
            "paper": "2310.11511.pdf" if index < 2 else "2312.10997.pdf",
            "page": index + 1,
            "score": 1.0 if index < 2 else 5.0,
        }
        for index in range(4)
    ]
    new = [
        {
            **sample_chunks[1],
            "chunk_id": f"crag-{index}",
            "paper": "2401.15884.pdf",
            "score": 2.0 - index,
        }
        for index in range(2)
    ]
    state = initial_state("Compare Self-RAG and CRAG")
    state["plan"] = {
        "queries": ["Self-RAG CRAG"],
        "entities": ["Self-RAG", "CRAG"],
        "targets": ["Self-RAG", "CRAG"],
        "facets": ["mechanism"],
        "output_language": "English",
    }
    state["evidence"] = old
    state["verification"]["corrective_query"] = "CRAG correction mechanism"

    def scored(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        return new

    result = researcher_node(
        state,
        FakeEngine(old + new),  # type: ignore[arg-type]
        Settings(),
        scored,
    )

    assert sum(target_matches("Self-RAG", item["text"]) for item in result["evidence"]) >= 2
    assert sum(target_matches("CRAG", item["text"]) for item in result["evidence"]) >= 2
    assert sum(item["paper"] == "2310.11511.pdf" for item in result["evidence"]) == 2
    assert result["retry_count"] == 1


def test_verifier_computes_coverage_and_rejects_target_mismatch(
    sample_chunks: list[dict],
) -> None:
    state = initial_state("Compare Self-RAG and CRAG")
    state["plan"].update(
        targets=["Self-RAG", "CRAG"],
        facets=["mechanism"],
    )
    state["evidence"] = sample_chunks[:2]
    assert verifier_node(state)["verification"]["status"] == "complete"

    mismatched = StubLLM(
        {
            "covered": {
                "Self-RAG": {"mechanism": ["E2"]},
                "CRAG": {"mechanism": ["E1", "E99"]},
            },
            "missing": [],
            "corrective_query": "",
        },
    )
    result = verifier_node(state, mismatched)  # type: ignore[arg-type]
    assert result["verification"]["status"] == "insufficient"


def test_writer_uses_only_covered_ids_and_abstains_without_citations(
    sample_chunks: list[dict],
) -> None:
    state = initial_state("Compare Self-RAG and CRAG")
    state["evidence"] = sample_chunks[:2]
    state["verification"] = {
        "status": "partial",
        "covered": {"Self-RAG": {"mechanism": ["E1"]}},
        "missing": ["CRAG: mechanism"],
        "corrective_query": "",
    }
    partial = writer_node(state)["answer"]
    assert "[Self-RAG.pdf p.1]" in partial
    assert "[CRAG.pdf p.2]" not in partial
    assert "Missing evidence" in partial

    state["verification"] = {
        "status": "insufficient",
        "covered": {},
        "missing": ["Self-RAG: mechanism", "CRAG: mechanism"],
        "corrective_query": "",
    }
    state["plan"]["output_language"] = "中文"
    abstention = writer_node(state)["answer"]
    assert "没有足够相关的证据" in abstention
    assert ".pdf p." not in abstention

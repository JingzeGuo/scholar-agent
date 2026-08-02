from __future__ import annotations

from typing import Any

from scholar_agent.agents.planner import planner_node, target_matches
from scholar_agent.agents.researcher import _select_evidence, researcher_node
from scholar_agent.agents.verifier import verifier_node
from scholar_agent.agents.writer import writer_node
from scholar_agent.config import Settings
from scholar_agent.workflow import initial_state


class StubLLM:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.last_prompt = ""

    def complete_json(self, prompt: str) -> dict[str, Any]:
        self.last_prompt = prompt
        if isinstance(self.payload, Exception):
            raise self.payload
        assert "Question" in prompt
        return self.payload  # type: ignore[return-value]


class FakeEngine:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.sparse_calls: list[list[str]] = []
        self.dense_calls: list[list[str]] = []

    def sparse_search(self, queries: list[str]) -> list[dict]:
        self.sparse_calls.append(queries)
        return self.chunks

    def dense_search(self, queries: list[str]) -> list[dict]:
        self.dense_calls.append(queries)
        return self.chunks

    def graph_search(self, entities: list[str]) -> list[dict]:
        return self.chunks


def test_planner_returns_compact_bounded_plan() -> None:
    payload = {
        "queries": ["q1", "q2", "q3", "q4"],
        "entities": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"],
        "targets": ["Alpha", "Beta", "Gamma", "Delta"],
        "facets": [
            "retrieval trigger",
            "key differences",
            "generation control",
            "evaluation results",
        ],
        "output_language": "Chinese",
    }
    llm = StubLLM(payload)
    plan = planner_node(
        initial_state("Compare Alpha, Beta, Gamma, and Delta"),
        llm,  # type: ignore[arg-type]
    )["plan"]

    assert plan["queries"] == ["q1", "q2", "q3"]
    assert plan["entities"] == ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    assert plan["targets"] == ["Alpha", "Beta", "Gamma"]
    assert plan["facets"] == payload["facets"]
    assert plan["output_language"] == "Chinese"
    assert "Do not invent targets" in llm.last_prompt
    assert "Do not invent requirements" in llm.last_prompt

    open_plan = planner_node(
        initial_state("Which retrieval methods are discussed in the corpus?"),
        StubLLM({**payload, "targets": ["Self-RAG", "CRAG"]}),  # type: ignore[arg-type]
    )["plan"]
    assert open_plan["targets"] == []
    assert open_plan["queries"] == ["q1", "q2", "q3"]
    assert open_plan["facets"] == payload["facets"]


def test_offline_planner_does_not_invent_facets() -> None:
    question = "Compare MethodA and MethodB"
    plan = planner_node(initial_state(question))["plan"]
    invalid_json_plan = planner_node(
        initial_state(question),
        StubLLM(ValueError("invalid JSON")),  # type: ignore[arg-type]
    )["plan"]

    assert plan["queries"] == [question]
    assert plan["facets"] == [question]
    assert plan["targets"] == ["MethodA", "MethodB"]
    assert invalid_json_plan == plan


def test_target_matching_preserves_method_identity() -> None:
    assert target_matches("Self-RAG", "Self RAG uses reflection tokens.")
    assert target_matches("CRAG", "CRAG uses a retrieval evaluator.")
    assert not target_matches("CRAG", "Self-CRAG combines both methods.")
    assert not target_matches("DPR", "ANCE uses one dense embedding.")
    assert not target_matches("RAG", "CRAG, Self-RAG, and GraphRAG are methods.")


def test_researcher_selection_uses_score_not_filename_age() -> None:
    items = [
        {
            "chunk_id": "older",
            "paper": "2020.1.pdf",
            "page": 1,
            "text": "MethodA retrieval evidence.",
            "score": 0.2,
        },
        {
            "chunk_id": "newer",
            "paper": "2025.9.pdf",
            "page": 1,
            "text": "MethodA retrieval evidence.",
            "score": 0.9,
        },
    ]

    selected = _select_evidence(items, [])

    assert selected[0]["score"] == 0.9


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

    state["plan"]["targets"] = ["Self-RAG", "RoseTTAFold All-Atom"]
    partial_target = researcher_node(
        state,
        FakeEngine(sample_chunks),  # type: ignore[arg-type]
        Settings(),
        high_scores,
    )
    assert partial_target["evidence"]
    assert partial_target["stop_reason"] == ""

    state["plan"]["targets"] = []
    open_question = researcher_node(
        state,
        FakeEngine(sample_chunks),  # type: ignore[arg-type]
        Settings(),
        high_scores,
    )
    assert open_question["evidence"]


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
            "page": index + 2,
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

    engine = FakeEngine(old + new)
    result = researcher_node(
        state,
        engine,  # type: ignore[arg-type]
        Settings(),
        scored,
    )

    assert sum(target_matches("Self-RAG", item["text"]) for item in result["evidence"]) >= 2
    assert sum(target_matches("CRAG", item["text"]) for item in result["evidence"]) >= 2
    assert sum(item["paper"] == "2310.11511.pdf" for item in result["evidence"]) == 2
    assert result["retry_count"] == 1
    assert all(len(call) == 1 for call in engine.sparse_calls + engine.dense_calls)
    assert {tuple(call) for call in engine.sparse_calls} == {
        ("Self-RAG CRAG",),
        ("CRAG correction mechanism",),
    }
    assert len({(item["paper"], item["page"]) for item in result["evidence"]}) == len(
        result["evidence"],
    )


def test_verifier_rejects_target_mismatch_and_invalid_ids(
    sample_chunks: list[dict],
) -> None:
    state = initial_state("Compare Self-RAG and CRAG")
    state["plan"].update(
        targets=["Self-RAG", "CRAG"],
        facets=["retrieval"],
    )
    state["evidence"] = sample_chunks[:2]
    assert verifier_node(state)["verification"]["status"] == "complete"

    mismatched = StubLLM(
        {
            "covered": {
                "Self-RAG": {"retrieval": ["E2"]},
                "CRAG": {"retrieval": ["E1", "E99"]},
            },
            "missing": [],
            "corrective_query": "",
        },
    )
    result = verifier_node(state, mismatched)  # type: ignore[arg-type]
    assert result["verification"]["status"] == "insufficient"


def test_verifier_rejects_shared_acronym_as_target_substitution(
    sample_chunks: list[dict],
) -> None:
    state = initial_state("Compare CRAG with Comprehensive RAG Benchmark.")
    state["plan"].update(
        targets=["CRAG", "Comprehensive RAG Benchmark"],
        facets=["identity"],
    )
    state["evidence"] = [
        sample_chunks[1],
        {
            **sample_chunks[2],
            "text": "Comprehensive RAG Benchmark (CRAG) is a benchmark dataset.",
        },
    ]
    ambiguous = verifier_node(
        state,
        StubLLM(
            {
                "covered": {
                    "CRAG": {"identity": ["E2"]},
                    "Comprehensive RAG Benchmark": {"identity": ["E2"]},
                },
                "corrective_query": "",
            },
        ),  # type: ignore[arg-type]
    )["verification"]
    assert ambiguous["status"] == "partial"
    assert ambiguous["covered"]["CRAG"] == {}


def test_verifier_returns_partial_for_missing_facet() -> None:
    state = initial_state("Explain MethodA retrieval and generation")
    state["plan"].update(
        targets=["MethodA"],
        facets=["retrieval", "generation"],
    )
    state["evidence"] = [
        {
            "chunk_id": "method-a",
            "paper": "MethodA.pdf",
            "page": 1,
            "text": "MethodA uses dynamic retrieval of passages.",
            "score": 1.0,
        },
    ]

    verification = verifier_node(state)["verification"]

    assert verification["status"] == "partial"
    assert verification["missing"] == ["MethodA: generation"]


def test_verifier_prefers_insufficient_to_false_coverage() -> None:
    state = initial_state("Explain MethodA retrieval")
    state["plan"].update(targets=["MethodA"], facets=["retrieval"])
    state["evidence"] = [
        {
            "chunk_id": "unrelated",
            "paper": "Other.pdf",
            "page": 1,
            "text": "This paragraph is unrelated.",
            "score": 1.0,
        },
    ]

    verification = verifier_node(state)["verification"]

    assert verification["status"] == "insufficient"
    assert verification["missing"] == ["MethodA: retrieval"]


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

from __future__ import annotations

from typing import Any

import pytest

from scholar_agent.agents.planner import planner_node, target_matches
from scholar_agent.agents.researcher import _select_evidence, researcher_node
from scholar_agent.agents.verifier import verifier_node
from scholar_agent.agents.writer import writer_node
from scholar_agent.config import Settings
from scholar_agent.workflow import initial_state


class StubLLM:
    def __init__(self, payload: object, text: str = "") -> None:
        self.payload = payload
        self.text = text
        self.last_prompt = ""

    def complete_json(self, prompt: str) -> dict[str, Any]:
        self.last_prompt = prompt
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload  # type: ignore[return-value]

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.text


def verifier_llm(covered: dict, corrective_query: str = "") -> StubLLM:
    return StubLLM({"covered": covered, "corrective_query": corrective_query})


def requirement(requirement_id: str, description: str, targets: list[str]) -> dict:
    return {"id": requirement_id, "description": description, "targets": targets}


class FakeEngine:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.sparse_calls: list[list[str]] = []
        self.dense_calls: list[list[str]] = []

    def sparse_search(self, queries: list[str]) -> list[dict]:
        self.sparse_calls.append(queries)
        return self.chunks

    def dense_search_many(self, queries: list[str]) -> list[list[dict]]:
        self.dense_calls.append(queries)
        return [self.chunks for _ in queries]


def test_planner_returns_compact_bounded_plan() -> None:
    payload = {
        "queries": ["q1", "q2", "q3", "q4", "q5", "q6"],
        "requirements": [
            {"description": "Explain Alpha's retrieval trigger", "targets": ["Alpha"]},
            {"description": "Identify Beta's limitations", "targets": ["Beta"]},
            {"description": "Compare Gamma and Delta", "targets": ["Gamma", "Delta"]},
            {"description": "Report Alpha's evaluation", "targets": ["Alpha"]},
            {"description": "Describe Beta's deployment", "targets": ["Beta"]},
            {"description": "Explain Gamma's generation", "targets": ["Gamma"]},
        ],
    }
    llm = StubLLM(payload)
    plan = planner_node(
        initial_state("Compare Alpha, Beta, Gamma, and Delta"),
        llm,  # type: ignore[arg-type]
    )["plan"]

    assert plan["queries"] == ["q1", "q2", "q3", "q4", "q5"]
    assert plan["requirements"] == [
        requirement("R1", "Explain Alpha's retrieval trigger", ["Alpha"]),
        requirement("R2", "Identify Beta's limitations", ["Beta"]),
        requirement("R3", "Compare Gamma and Delta", ["Gamma", "Delta"]),
        requirement("R4", "Report Alpha's evaluation", ["Alpha"]),
        requirement("R5", "Describe Beta's deployment", ["Beta"]),
    ]
    assert set(plan) == {"queries", "requirements"}
    assert "plan retrieval and verification" in llm.last_prompt
    assert "do not answer the question" in llm.last_prompt
    assert 'Every "requirement" is one independent' in llm.last_prompt
    assert "both BM25 and dense retrieval" in llm.last_prompt
    assert "Do not invent targets or requirements" in llm.last_prompt
    assert "Keep asymmetric requests separate" in llm.last_prompt
    assert "<user_question>" in llm.last_prompt

    open_plan = planner_node(
        initial_state("Which retrieval methods are discussed in the corpus?"),
        StubLLM(
            {
                **payload,
                "requirements": [
                    {
                        "description": "Identify the retrieval methods discussed",
                        "targets": [],
                    },
                ],
            },
        ),  # type: ignore[arg-type]
    )["plan"]
    assert open_plan["queries"] == ["q1", "q2", "q3", "q4", "q5"]
    assert open_plan["requirements"] == [
        requirement("R1", "Identify the retrieval methods discussed", []),
    ]


def test_planner_rejects_invalid_llm_output() -> None:
    question = "Compare MethodA and MethodB"

    with pytest.raises(ValueError, match="invalid JSON"):
        planner_node(
            initial_state(question),
            StubLLM(ValueError("invalid JSON")),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="no queries"):
        planner_node(
            initial_state(question),
            StubLLM(
                {
                    "queries": [],
                    "requirements": [
                        {"description": "Compare the methods", "targets": ["MethodA", "MethodB"]},
                    ],
                },
            ),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="no valid requirements"):
        planner_node(
            initial_state(question),
            StubLLM(
                {
                    "queries": ["MethodA MethodB"],
                    "requirements": [],
                },
            ),  # type: ignore[arg-type]
        )


def test_planner_preserves_asymmetric_atomic_requirements() -> None:
    state = initial_state("Explain Self-RAG retrieval triggers and CRAG limitations")
    plan = planner_node(
        state,
        StubLLM(
            {
                "queries": ["Self-RAG retrieval triggers", "CRAG limitations"],
                "requirements": [
                    {
                        "description": "Explain Self-RAG retrieval triggers",
                        "targets": ["Self-RAG"],
                    },
                    {
                        "description": "Identify CRAG limitations",
                        "targets": ["CRAG"],
                    },
                ],
            },
        ),  # type: ignore[arg-type]
    )["plan"]

    assert plan["requirements"] == [
        requirement("R1", "Explain Self-RAG retrieval triggers", ["Self-RAG"]),
        requirement("R2", "Identify CRAG limitations", ["CRAG"]),
    ]


def test_target_matching_preserves_method_identity() -> None:
    assert target_matches("Self-RAG", "Self RAG uses reflection tokens.")
    assert target_matches("CRAG", "CRAG uses a retrieval evaluator.")
    assert not target_matches("CRAG", "Self-CRAG combines both methods.")
    assert not target_matches("DPR", "ANCE uses one dense embedding.")
    assert not target_matches("RAG", "CRAG and Self-RAG are methods.")


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


def test_researcher_runs_bm25_and_dense_for_every_query(sample_chunks: list[dict]) -> None:
    state = initial_state("Explain Self-RAG")
    state["plan"].update(
        queries=["Self-RAG retrieval", "reflection tokens"],
        requirements=[requirement("R1", "Explain Self-RAG's mechanism", ["Self-RAG"])],
    )
    rerank_inputs: list[list[str]] = []

    def scored(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        rerank_inputs.append([item["chunk_id"] for item in candidates])
        return [{**item, "score": 1.0} for item in candidates]

    engine = FakeEngine(sample_chunks[:2])
    result = researcher_node(
        state,
        engine,  # type: ignore[arg-type]
        Settings(),
        scored,
    )

    assert engine.sparse_calls == [["Self-RAG retrieval"], ["reflection tokens"]]
    assert engine.dense_calls == [["Self-RAG retrieval", "reflection tokens"]]
    assert rerank_inputs == [["self-1", "crag-1"]]
    assert [item["chunk_id"] for item in result["evidence"]] == ["self-1", "crag-1"]


def test_researcher_deduplicates_physical_pages(sample_chunks: list[dict]) -> None:
    duplicate_page = {
        **sample_chunks[0],
        "chunk_id": "self-2",
        "text": "Self-RAG also uses reflection tokens.",
        "score": 0.8,
    }

    selected = _select_evidence(
        [{**sample_chunks[0], "score": 0.9}, duplicate_page, sample_chunks[1]],
        [],
    )

    assert [item["chunk_id"] for item in selected] == ["self-1", "crag-1"]


def test_researcher_rejects_every_below_threshold_chunk(sample_chunks: list[dict]) -> None:
    state = initial_state("Compare Self-RAG and CRAG")
    state["plan"] = {
        "queries": ["Self-RAG CRAG"],
        "requirements": [
            requirement("R1", "Explain Self-RAG's mechanism", ["Self-RAG"]),
            requirement("R2", "Explain CRAG's mechanism", ["CRAG"]),
        ],
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

    state["plan"]["requirements"] = [
        requirement(
            "R1",
            "Explain RoseTTAFold All-Atom",
            ["RoseTTAFold All-Atom"],
        ),
    ]

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

    state["plan"]["requirements"] = [
        requirement("R1", "Explain Self-RAG", ["Self-RAG"]),
        requirement(
            "R2",
            "Explain RoseTTAFold All-Atom",
            ["RoseTTAFold All-Atom"],
        ),
    ]
    partial_target = researcher_node(
        state,
        FakeEngine(sample_chunks),  # type: ignore[arg-type]
        Settings(),
        high_scores,
    )
    assert partial_target["evidence"]
    assert partial_target["stop_reason"] == ""

    state["plan"]["requirements"] = [
        requirement("R1", "Identify relevant methods", []),
    ]
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
        "requirements": [
            requirement("R1", "Explain Self-RAG's mechanism", ["Self-RAG"]),
            requirement("R2", "Explain CRAG's mechanism", ["CRAG"]),
        ],
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
    assert engine.sparse_calls == [["CRAG correction mechanism"]]
    assert engine.dense_calls == [["CRAG correction mechanism"]]
    assert len({(item["paper"], item["page"]) for item in result["evidence"]}) == len(
        result["evidence"],
    )


def test_researcher_marks_identical_retry_evidence(sample_chunks: list[dict]) -> None:
    state = initial_state("Explain Self-RAG")
    state["plan"].update(
        queries=["Self-RAG"],
        requirements=[requirement("R1", "Explain Self-RAG's mechanism", ["Self-RAG"])],
    )
    state["evidence"] = [{**sample_chunks[0], "score": 1.0}]
    state["verification"]["corrective_query"] = "Self-RAG mechanism"

    def scored(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        return [{**item, "score": 1.0} for item in candidates]

    result = researcher_node(
        state,
        FakeEngine(sample_chunks[:1]),  # type: ignore[arg-type]
        Settings(),
        scored,
    )

    assert result["evidence"] == state["evidence"]
    assert result["retry_count"] == 1
    assert result["stop_reason"] == "no_new_evidence"


def test_verifier_rejects_target_mismatch_and_invalid_ids(
    sample_chunks: list[dict],
) -> None:
    state = initial_state("Compare Self-RAG and CRAG")
    state["plan"].update(
        requirements=[
            requirement("R1", "Explain Self-RAG retrieval", ["Self-RAG"]),
            requirement("R2", "Explain CRAG retrieval", ["CRAG"]),
        ],
    )
    state["evidence"] = sample_chunks[:2]
    complete = verifier_llm(
        {
            "R1": ["E1"],
            "R2": ["E2"],
        },
    )
    assert verifier_node(state, complete)["verification"]["status"] == "complete"  # type: ignore[arg-type]

    mismatched = StubLLM(
        {
            "covered": {
                "R1": ["E2"],
                "R2": ["E1", "E99"],
            },
            "missing": [],
            "corrective_query": "",
        },
    )
    result = verifier_node(state, mismatched)  # type: ignore[arg-type]
    assert result["verification"]["status"] == "insufficient"

    with pytest.raises(ValueError, match="covered must be an object"):
        verifier_node(state, StubLLM({"covered": None}))  # type: ignore[arg-type]


def test_verifier_rejects_shared_acronym_as_target_substitution(
    sample_chunks: list[dict],
) -> None:
    state = initial_state("Compare CRAG with Comprehensive RAG Benchmark.")
    state["plan"].update(
        requirements=[
            requirement("R1", "Identify CRAG", ["CRAG"]),
            requirement(
                "R2",
                "Identify Comprehensive RAG Benchmark",
                ["Comprehensive RAG Benchmark"],
            ),
        ],
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
                    "R1": ["E2"],
                    "R2": ["E2"],
                },
                "corrective_query": "",
            },
        ),  # type: ignore[arg-type]
    )["verification"]
    assert ambiguous["status"] == "partial"
    assert "R1" not in ambiguous["covered"]
    assert ambiguous["covered"]["R2"] == ["E2"]


def test_verifier_requires_evidence_for_every_target_in_one_requirement(
    sample_chunks: list[dict],
) -> None:
    state = initial_state("Compare Self-RAG and CRAG retrieval")
    state["plan"]["requirements"] = [
        requirement(
            "R1",
            "Compare Self-RAG and CRAG retrieval",
            ["Self-RAG", "CRAG"],
        ),
    ]
    state["evidence"] = sample_chunks[:2]

    partial = verifier_node(
        state,
        verifier_llm({"R1": ["E1"]}),  # type: ignore[arg-type]
    )["verification"]
    complete = verifier_node(
        state,
        verifier_llm({"R1": ["E1", "E2"]}),  # type: ignore[arg-type]
    )["verification"]

    assert partial["status"] == "insufficient"
    assert partial["missing"] == ["R1"]
    assert complete["status"] == "complete"
    assert complete["covered"] == {"R1": ["E1", "E2"]}


def test_verifier_returns_partial_for_missing_requirement() -> None:
    state = initial_state("Explain MethodA retrieval and generation")
    state["plan"].update(
        requirements=[
            requirement("R1", "Explain MethodA retrieval", ["MethodA"]),
            requirement("R2", "Explain MethodA generation", ["MethodA"]),
        ],
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

    verification = verifier_node(
        state,
        verifier_llm({"R1": ["E1"]}),  # type: ignore[arg-type]
    )["verification"]

    assert verification["status"] == "partial"
    assert verification["missing"] == ["R2"]


def test_verifier_prefers_insufficient_to_false_coverage() -> None:
    state = initial_state("Explain MethodA retrieval")
    state["plan"].update(
        requirements=[requirement("R1", "Explain MethodA retrieval", ["MethodA"])],
    )
    state["evidence"] = [
        {
            "chunk_id": "unrelated",
            "paper": "Other.pdf",
            "page": 1,
            "text": "This paragraph is unrelated.",
            "score": 1.0,
        },
    ]

    verification = verifier_node(state, verifier_llm({}))["verification"]  # type: ignore[arg-type]

    assert verification["status"] == "insufficient"
    assert verification["missing"] == ["R1"]


def test_writer_uses_only_covered_ids_and_abstains_without_citations(
    sample_chunks: list[dict],
) -> None:
    state = initial_state("Compare Self-RAG and CRAG")
    state["plan"]["requirements"] = [
        requirement("R1", "Explain Self-RAG's mechanism", ["Self-RAG"]),
        requirement("R2", "Explain CRAG's mechanism", ["CRAG"]),
    ]
    state["evidence"] = sample_chunks[:2]
    state["verification"] = {
        "status": "partial",
        "covered": {"R1": ["E1"]},
        "missing": ["R2"],
        "corrective_query": "",
    }
    partial = writer_node(
        state,
        StubLLM(
            {},
            "Self-RAG uses adaptive retrieval [E1]. Missing evidence: CRAG mechanism.",
        ),  # type: ignore[arg-type]
    )["answer"]
    assert "[Self-RAG.pdf p.1]" in partial
    assert "[CRAG.pdf p.2]" not in partial
    assert "Missing evidence" in partial

    state["verification"] = {
        "status": "insufficient",
        "covered": {},
        "missing": ["R1", "R2"],
        "corrective_query": "",
    }
    llm = StubLLM({}, "The corpus does not contain enough relevant evidence.")
    abstention = writer_node(state, llm)["answer"]  # type: ignore[arg-type]
    assert "enough relevant evidence" in abstention
    assert ".pdf p." not in abstention
    assert "Answer in English" in llm.last_prompt


def test_writer_always_requests_english() -> None:
    state = initial_state("Cette preuve existe-t-elle ?")
    llm = StubLLM({}, "The corpus does not contain sufficiently relevant evidence.")

    answer = writer_node(state, llm)["answer"]  # type: ignore[arg-type]

    assert answer == "The corpus does not contain sufficiently relevant evidence."
    assert "Answer in English" in llm.last_prompt
    assert "Answer in French" not in llm.last_prompt
    assert "Status: insufficient" in llm.last_prompt


def test_writer_rejects_unverified_citations(sample_chunks: list[dict]) -> None:
    state = initial_state("Explain Self-RAG")
    state["plan"]["requirements"] = [
        requirement("R1", "Explain Self-RAG's mechanism", ["Self-RAG"]),
    ]
    state["evidence"] = sample_chunks[:2]
    state["verification"] = {
        "status": "complete",
        "covered": {"R1": ["E1"]},
        "missing": [],
        "corrective_query": "",
    }

    with pytest.raises(ValueError, match="not approved"):
        writer_node(state, StubLLM({}, "Unsupported [E2]."))  # type: ignore[arg-type]

    state["verification"] = {
        "status": "insufficient",
        "covered": {},
        "missing": ["R1"],
        "corrective_query": "",
    }
    with pytest.raises(ValueError, match="while abstaining"):
        writer_node(state, StubLLM({}, "Unsupported [E1]."))  # type: ignore[arg-type]

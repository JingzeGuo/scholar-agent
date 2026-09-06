from __future__ import annotations

from typing import Any

import pytest

from scholar_agent.agents.planner import evidence_matches_target, planner_node, target_matches
from scholar_agent.agents.researcher import (
    _attach_requirement_scores,
    _planned_queries,
    _select_candidates_for_reranking,
    _select_evidence,
    researcher_node,
)
from scholar_agent.agents.verifier import verifier_node
from scholar_agent.agents.writer import SAFE_ABSTENTION, writer_node
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


class SequenceLLM(StubLLM):
    def __init__(self, texts: list[str]) -> None:
        super().__init__({})
        self.texts = iter(texts)

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return next(self.texts)


def verifier_llm(
    covered: dict,
    corrective_queries: list[dict] | None = None,
    uncertain: dict | None = None,
) -> StubLLM:
    return StubLLM(
        {
            "covered": covered,
            "uncertain": uncertain or {},
            "corrective_queries": corrective_queries or [],
        },
    )


def requirement(
    requirement_id: str,
    description: str,
    targets: list[str],
    query: str | None = None,
) -> dict:
    result = {"id": requirement_id, "description": description, "targets": targets}
    if query is not None:
        result["query"] = query
    return result


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
        "requirements": [
            {
                "description": "Explain Alpha's retrieval trigger",
                "targets": ["Alpha"],
                "query": "q1",
            },
            {"description": "Identify Beta's limitations", "targets": ["Beta"], "query": "q2"},
            {
                "description": "Compare Gamma and Delta",
                "targets": ["Gamma", "Delta"],
                "query": "q3",
            },
            {"description": "Report Alpha's evaluation", "targets": ["Alpha"], "query": "q4"},
            {"description": "Describe Beta's deployment", "targets": ["Beta"], "query": "q5"},
            {"description": "Explain Gamma's generation", "targets": ["Gamma"], "query": "q6"},
        ],
    }
    llm = StubLLM(payload)
    plan = planner_node(
        initial_state("Compare Alpha, Beta, Gamma, and Delta"),
        llm,  # type: ignore[arg-type]
    )["plan"]

    assert plan["requirements"] == [
        requirement("R1", "Explain Alpha's retrieval trigger", ["Alpha"], "q1"),
        requirement("R2", "Identify Beta's limitations", ["Beta"], "q2"),
        requirement("R3", "Compare Gamma and Delta", ["Gamma", "Delta"], "q3"),
        requirement("R4", "Report Alpha's evaluation", ["Alpha"], "q4"),
        requirement("R5", "Describe Beta's deployment", ["Beta"], "q5"),
    ]
    assert set(plan) == {"requirements"}
    assert "plan retrieval and verification" in llm.last_prompt
    assert "do not answer the question" in llm.last_prompt
    assert 'Every "requirement" is one independent' in llm.last_prompt
    assert "both BM25 and dense retrieval" in llm.last_prompt
    assert "Do not invent targets or requirements" in llm.last_prompt
    assert "Keep asymmetric requests separate" in llm.last_prompt
    assert "comparison can be synthesized" in llm.last_prompt
    assert "Copy each target exactly" in llm.last_prompt
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
                        "query": "retrieval methods discussed in the corpus",
                    },
                ],
            },
        ),  # type: ignore[arg-type]
    )["plan"]
    assert open_plan["requirements"] == [
        requirement(
            "R1",
            "Identify the retrieval methods discussed",
            [],
            "retrieval methods discussed in the corpus",
        ),
    ]


def test_planner_falls_back_to_the_question_for_invalid_llm_output() -> None:
    question = "Compare MethodA and MethodB"
    invalid_outputs = [
        ValueError("invalid JSON"),
        {
            "requirements": [
                {
                    "description": "Compare the methods",
                    "targets": ["MethodA", "MethodB"],
                    "query": "",
                },
            ],
        },
        {"requirements": []},
    ]

    for output in invalid_outputs:
        plan = planner_node(
            initial_state(question),
            StubLLM(output),  # type: ignore[arg-type]
        )["plan"]
        assert plan["requirements"] == [
            requirement("R1", question, [], question),
        ]


def test_planner_preserves_asymmetric_atomic_requirements() -> None:
    state = initial_state("Explain Self-RAG retrieval triggers and CRAG limitations")
    plan = planner_node(
        state,
        StubLLM(
            {
                "requirements": [
                    {
                        "description": "Explain Self-RAG retrieval triggers",
                        "targets": ["Self-RAG"],
                        "query": "Self-RAG retrieval triggers",
                    },
                    {
                        "description": "Identify CRAG limitations",
                        "targets": ["CRAG"],
                        "query": "CRAG limitations",
                    },
                ],
            },
        ),  # type: ignore[arg-type]
    )["plan"]

    assert plan["requirements"] == [
        requirement(
            "R1",
            "Explain Self-RAG retrieval triggers",
            ["Self-RAG"],
            "Self-RAG retrieval triggers",
        ),
        requirement("R2", "Identify CRAG limitations", ["CRAG"], "CRAG limitations"),
    ]


def test_target_matching_preserves_method_identity() -> None:
    assert target_matches("Self-RAG", "Self RAG uses reflection tokens.")
    assert target_matches("CRAG", "CRAG uses a retrieval evaluator.")
    assert not target_matches("CRAG", "Self-CRAG combines both methods.")
    assert not target_matches("DPR", "ANCE uses one dense embedding.")
    assert not target_matches("RAG", "CRAG and Self-RAG are methods.")
    assert evidence_matches_target(
        "Self-RAG",
        {"paper": "Self-RAG.pdf", "text": "The proposed method retrieves passages."},
    )


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
        requirements=[
            requirement(
                "R1",
                "Explain Self-RAG's retrieval",
                ["Self-RAG"],
                "Self-RAG retrieval",
            ),
            requirement(
                "R2",
                "Explain Self-RAG's reflection tokens",
                ["Self-RAG"],
                "reflection tokens",
            ),
        ],
    )
    rerank_inputs: list[list[str]] = []

    def scored(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        rerank_inputs.append([item["chunk_id"] for item in candidates])
        return [
            {**item, "score": 1.0, "_query_scores": [1.0] * len(queries)}
            for item in candidates
        ]

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


def test_researcher_keeps_one_query_per_requirement() -> None:
    plan = {
        "requirements": [
            requirement(
                "R1",
                "Explain Self-RAG's retrieval mechanism",
                ["Self-RAG"],
                "Self-RAG retrieval mechanism",
            ),
            requirement(
                "R2",
                "Describe when Self-RAG retrieves",
                ["Self-RAG"],
                "  self-rag   retrieval mechanism  ",
            ),
        ],
    }

    queries, query_requirement_ids = _planned_queries(plan)

    assert queries == [
        "Self-RAG retrieval mechanism",
        "self-rag   retrieval mechanism",
    ]
    assert query_requirement_ids == [["R1"], ["R2"]]


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


def test_researcher_allows_same_page_and_paper_for_requirement_coverage() -> None:
    requirements = [
        requirement(f"R{index}", f"Explain aspect {index}", []) for index in range(1, 6)
    ]
    items = [
        {
            "chunk_id": f"aspect-{index}",
            "paper": "Method.pdf",
            "page": min(index, 4),
            "text": f"Evidence for aspect {index}.",
            "score": 1.0,
            "_requirement_scores": {
                requirement["id"]: 1.0 if requirement["id"] == f"R{index}" else 0.0
                for requirement in requirements
            },
        }
        for index in range(1, 6)
    ]

    selected = _select_evidence(items, requirements, min_score=0.5)

    assert {item["chunk_id"] for item in selected} == {
        "aspect-1",
        "aspect-2",
        "aspect-3",
        "aspect-4",
        "aspect-5",
    }


def test_researcher_expands_evidence_limit_for_multi_target_coverage() -> None:
    requirements = [
        requirement(
            f"R{requirement_index}",
            f"Compare group {requirement_index}",
            [f"Target-{requirement_index}-{target_index}" for target_index in range(1, 4)],
        )
        for requirement_index in range(1, 4)
    ]
    items = [
        {
            "chunk_id": target,
            "paper": f"{target}.pdf",
            "page": 1,
            "text": f"{target} evidence.",
            "score": 1.0,
            "_requirement_scores": {
                requirement["id"]: 1.0 if target in requirement["targets"] else 0.0
                for requirement in requirements
            },
        }
        for requirement in requirements
        for target in requirement["targets"]
    ]

    selected = _select_evidence(items, requirements, min_score=0.5)

    assert len(selected) == 9


def test_researcher_does_not_reuse_a_retry_score_for_unscored_requirements() -> None:
    requirements = [
        requirement("R1", "Explain the mechanism", []),
        requirement("R2", "Explain the limitations", []),
    ]
    mechanism = {
        "chunk_id": "mechanism",
        "paper": "Method.pdf",
        "page": 1,
        "text": "Mechanism evidence.",
        "score": 0.6,
        "_requirement_scores": {"R1": 0.6, "R2": 0.0},
    }
    retry_items = [
        {
            "chunk_id": f"limitation-{index}",
            "paper": f"Limitations-{index}.pdf",
            "page": 1,
            "text": f"Limitation evidence {index}.",
            "score": 0.99 - index / 100,
            "_requirement_scores": {"R2": 0.99 - index / 100},
        }
        for index in range(8)
    ]

    selected = _select_evidence([mechanism, *retry_items], requirements, min_score=0.5)

    assert "mechanism" in {item["chunk_id"] for item in selected}


def test_researcher_requires_per_query_reranker_scores() -> None:
    with pytest.raises(ValueError, match="one query score per query"):
        _attach_requirement_scores(
            [{"chunk_id": "c1", "score": 1.0}],
            [["R1"], ["R2"]],
        )


def test_researcher_reserves_evidence_for_requirements_with_the_same_target() -> None:
    requirements = [
        requirement(
            "R1",
            "Explain Self-RAG's retrieval mechanism",
            ["Self-RAG"],
            "Self-RAG retrieval mechanism",
        ),
        requirement(
            "R2",
            "Identify Self-RAG's limitations",
            ["Self-RAG"],
            "Self-RAG limitations",
        ),
    ]
    mechanism_items = [
        {
            "chunk_id": f"mechanism-{index}",
            "paper": f"Mechanism-{index}.pdf",
            "page": 1,
            "text": f"Self-RAG retrieval mechanism detail {index}.",
            "score": 0.99 - index / 100,
            "_requirement_scores": {
                "R1": 0.99 - index / 100,
                "R2": 0.1,
            },
        }
        for index in range(8)
    ]
    limitation = {
        "chunk_id": "limitation",
        "paper": "Limitations.pdf",
        "page": 1,
        "text": "Self-RAG limitations and failure cases.",
        "score": 0.9,
        "_requirement_scores": {"R1": 0.1, "R2": 0.9},
    }

    selected = _select_evidence(mechanism_items + [limitation], requirements, min_score=0.5)

    assert len(selected) == 8
    assert "limitation" in {item["chunk_id"] for item in selected}


def test_researcher_selects_reranking_candidates_for_every_query() -> None:
    sparse_rankings: list[list[dict]] = []
    dense_rankings: list[list[dict]] = []
    for query_index in range(4):
        shared = [
            {
                "chunk_id": f"frequent-{query_index}-{rank}",
                "paper": "Frequent.pdf",
                "page": rank + 1,
                "text": "frequent evidence",
                "score": 0.0,
            }
            for rank in range(8)
        ]
        sparse_rankings.append(shared)
        dense_rankings.append(shared)

    sparse_rankings.append(
        sparse_rankings[0][:4]
        + [
            {
                "chunk_id": f"rare-{rank}",
                "paper": "Rare.pdf",
                "page": rank + 1,
                "text": "rare requirement evidence",
                "score": 0.0,
            }
            for rank in range(4)
        ],
    )
    dense_rankings.append([])

    candidates = _select_candidates_for_reranking(sparse_rankings, dense_rankings)

    assert len(candidates) == 30
    assert "rare-0" in {item["chunk_id"] for item in candidates}


def test_researcher_accepts_target_identity_from_paper_name() -> None:
    item = {
        "chunk_id": "method-a",
        "paper": "MethodA.pdf",
        "page": 3,
        "text": "The proposed method retrieves passages dynamically.",
        "score": 0.9,
        "_requirement_scores": {"R1": 0.9},
    }

    selected = _select_evidence(
        [item],
        [requirement("R1", "Explain MethodA retrieval", ["MethodA"])],
        min_score=0.5,
    )

    assert selected == [item]


def test_researcher_rejects_every_below_threshold_chunk(sample_chunks: list[dict]) -> None:
    state = initial_state("Compare Self-RAG and CRAG")
    state["plan"] = {
        "requirements": [
            requirement(
                "R1",
                "Explain Self-RAG's mechanism",
                ["Self-RAG"],
                "Self-RAG CRAG",
            ),
            requirement("R2", "Explain CRAG's mechanism", ["CRAG"], "Self-RAG CRAG"),
        ],
    }

    def low_scores(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        return [
            {**item, "score": 0.4, "_query_scores": [0.4] * len(queries)}
            for item in candidates
        ]

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
            "RoseTTAFold All-Atom",
        ),
    ]

    def high_scores(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        return [
            {**item, "score": 9.0, "_query_scores": [9.0] * len(queries)}
            for item in candidates
        ]

    absent_target = researcher_node(
        state,
        FakeEngine(sample_chunks),  # type: ignore[arg-type]
        Settings(),
        high_scores,
    )
    assert absent_target["evidence"]
    assert absent_target["stop_reason"] == ""

    state["plan"]["requirements"] = [
        requirement("R1", "Explain Self-RAG", ["Self-RAG"], "Self-RAG CRAG"),
        requirement(
            "R2",
            "Explain RoseTTAFold All-Atom",
            ["RoseTTAFold All-Atom"],
            "Self-RAG CRAG",
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
        requirement("R1", "Identify relevant methods", [], "Self-RAG CRAG"),
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
        "requirements": [
            requirement("R1", "Explain Self-RAG's mechanism", ["Self-RAG"]),
            requirement("R2", "Explain CRAG's mechanism", ["CRAG"]),
        ],
    }
    state["evidence"] = old
    state["verification"]["missing"] = ["R2"]
    state["verification"]["corrective_queries"] = [
        {"requirement_id": "R2", "query": "CRAG correction mechanism"},
    ]

    def scored(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        return [{**item, "_query_scores": [item["score"]]} for item in new]

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
    assert next(
        item for item in result["evidence"] if item["chunk_id"] == "crag-0"
    )["_requirement_scores"] == {"R2": 2.0}
    assert len({(item["paper"], item["page"]) for item in result["evidence"]}) == len(
        result["evidence"],
    )


def test_researcher_marks_identical_retry_evidence(sample_chunks: list[dict]) -> None:
    state = initial_state("Explain Self-RAG")
    state["plan"].update(
        requirements=[requirement("R1", "Explain Self-RAG's mechanism", ["Self-RAG"])],
    )
    state["evidence"] = [{**sample_chunks[0], "score": 1.0}]
    state["verification"]["missing"] = ["R1"]
    state["verification"]["corrective_queries"] = [
        {"requirement_id": "R1", "query": "Self-RAG mechanism"},
    ]

    def scored(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        return [
            {**item, "score": 1.0, "_query_scores": [1.0] * len(queries)}
            for item in candidates
        ]

    result = researcher_node(
        state,
        FakeEngine(sample_chunks[:1]),  # type: ignore[arg-type]
        Settings(),
        scored,
    )

    assert result["evidence"] == state["evidence"]
    assert result["retry_count"] == 1
    assert result["stop_reason"] == "no_new_evidence"


def test_researcher_batches_corrective_queries(sample_chunks: list[dict]) -> None:
    state = initial_state("Compare Self-RAG and CRAG")
    state["plan"]["requirements"] = [
        requirement("R1", "Explain Self-RAG retrieval", ["Self-RAG"]),
        requirement("R2", "Explain CRAG retrieval", ["CRAG"]),
    ]
    state["verification"].update(
        missing=["R1", "R2"],
        corrective_queries=[
            {"requirement_id": "R1", "query": "Self-RAG retrieval evidence"},
            {"requirement_id": "R2", "query": "CRAG retrieval evidence"},
        ],
    )

    def scored(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        return [
            {
                **item,
                "score": 2.0,
                "_query_scores": [2.0, -1.0]
                if item["chunk_id"] == "self-1"
                else [-1.0, 2.0],
            }
            for item in candidates
        ]

    engine = FakeEngine(sample_chunks[:2])
    result = researcher_node(
        state,
        engine,  # type: ignore[arg-type]
        Settings(),
        scored,
    )

    assert engine.sparse_calls == [
        ["Self-RAG retrieval evidence"],
        ["CRAG retrieval evidence"],
    ]
    assert engine.dense_calls == [
        ["Self-RAG retrieval evidence", "CRAG retrieval evidence"],
    ]
    assert result["retry_count"] == 1
    assert {item["chunk_id"] for item in result["evidence"]} == {"self-1", "crag-1"}


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
            "corrective_queries": [],
        },
    )
    result = verifier_node(state, mismatched)  # type: ignore[arg-type]
    assert result["verification"]["status"] == "insufficient"

    malformed = verifier_node(
        state,
        StubLLM({"covered": None}),  # type: ignore[arg-type]
    )["verification"]
    assert malformed["status"] == "insufficient"


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
                "corrective_queries": [],
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


def test_verifier_preserves_uncertain_evidence_as_advice() -> None:
    state = initial_state("Explain MethodA retrieval")
    state["plan"]["requirements"] = [
        requirement("R1", "Explain MethodA retrieval", ["MethodA"]),
    ]
    state["evidence"] = [
        {
            "chunk_id": "method-a",
            "paper": "MethodA.pdf",
            "page": 1,
            "text": "MethodA may retrieve passages dynamically.",
            "score": 1.0,
        },
    ]

    verification = verifier_node(
        state,
        verifier_llm({}, uncertain={"R1": ["E1"]}),  # type: ignore[arg-type]
    )["verification"]

    assert verification["status"] == "partial"
    assert verification["covered"] == {}
    assert verification["uncertain"] == {"R1": ["E1"]}
    assert verification["missing"] == []


def test_verifier_batches_queries_for_missing_requirements() -> None:
    state = initial_state("Explain MethodA retrieval, generation, and evaluation")
    state["plan"]["requirements"] = [
        requirement("R1", "Explain MethodA retrieval", ["MethodA"]),
        requirement("R2", "Explain MethodA generation", ["MethodA"]),
        requirement("R3", "Explain MethodA evaluation", ["MethodA"]),
    ]
    state["evidence"] = [
        {
            "chunk_id": "retrieval",
            "paper": "MethodA.pdf",
            "page": 1,
            "text": "The proposed method retrieves passages dynamically.",
            "score": 1.0,
        },
    ]

    verification = verifier_node(
        state,
        verifier_llm(
            {"R1": ["E1"]},
            [
                {"requirement_id": "r3", "query": "MethodA evaluation evidence"},
                {"requirement_id": "r2", "query": "MethodA generation evidence"},
            ],
        ),  # type: ignore[arg-type]
    )["verification"]

    assert verification["corrective_queries"] == [
        {"requirement_id": "R2", "query": "MethodA generation evidence"},
        {"requirement_id": "R3", "query": "MethodA evaluation evidence"},
    ]

    malformed = verifier_node(
        state,
        verifier_llm(
            {"R1": ["E1"]},
            [{"requirement_id": "R1", "query": "MethodA generation evidence"}],
        ),  # type: ignore[arg-type]
    )["verification"]
    assert malformed["corrective_queries"] == []


def test_verifier_accepts_target_identity_from_paper_name() -> None:
    state = initial_state("Explain MethodA retrieval")
    state["plan"]["requirements"] = [
        requirement("R1", "Explain MethodA retrieval", ["MethodA"]),
    ]
    state["evidence"] = [
        {
            "chunk_id": "method-a",
            "paper": "MethodA.pdf",
            "page": 1,
            "text": "The proposed method retrieves passages dynamically.",
            "score": 1.0,
        },
    ]

    llm = verifier_llm({"R1": ["E1"]})
    verification = verifier_node(state, llm)["verification"]  # type: ignore[arg-type]

    assert verification["status"] == "complete"
    assert (
        "E1 [MethodA.pdf p.1]: The proposed method retrieves passages dynamically."
        in llm.last_prompt
    )


def test_verifier_accepts_semantic_support_when_the_target_name_is_not_repeated() -> None:
    state = initial_state("Explain MethodA retrieval")
    state["plan"]["requirements"] = [
        requirement("R1", "Explain MethodA retrieval", ["MethodA"]),
    ]
    state["evidence"] = [
        {
            "chunk_id": "method-a-title",
            "paper": "1234.56789.pdf",
            "page": 1,
            "text": "MethodA is a retrieval system.",
            "score": 1.0,
        },
        {
            "chunk_id": "method-a-detail",
            "paper": "1234.56789.pdf",
            "page": 4,
            "text": "The proposed approach retrieves passages dynamically.",
            "score": 1.0,
        },
    ]

    verification = verifier_node(
        state,
        verifier_llm({"R1": ["E2"]}),  # type: ignore[arg-type]
    )["verification"]

    assert verification["status"] == "complete"
    assert verification["covered"] == {"R1": ["E2"]}


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


def test_writer_sees_all_evidence_and_treats_coverage_as_advisory(
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
        "uncertain": {"R2": ["E2"]},
        "missing": [],
        "corrective_queries": [],
    }
    partial = writer_node(
        state,
        StubLLM(
            {},
            "Self-RAG uses adaptive retrieval [E1]. CRAG uses correction [E2].",
        ),  # type: ignore[arg-type]
    )["answer"]
    assert "[Self-RAG.pdf p.1]" in partial
    assert "[CRAG.pdf p.2]" in partial

    state["verification"] = {
        "status": "insufficient",
        "covered": {},
        "uncertain": {},
        "missing": ["R1", "R2"],
        "corrective_queries": [],
    }
    llm = StubLLM({}, SAFE_ABSTENTION)
    abstention = writer_node(state, llm)["answer"]  # type: ignore[arg-type]
    assert abstention == SAFE_ABSTENTION
    assert ".pdf p." not in abstention
    assert "[E1]" in llm.last_prompt
    assert "[E2]" in llm.last_prompt


def test_writer_always_requests_english() -> None:
    state = initial_state("Cette preuve existe-t-elle ?")
    state["plan"]["requirements"] = [
        requirement("R1", "Explain the evidence", [], "evidence"),
    ]
    state["evidence"] = [
        {
            "chunk_id": "evidence",
            "paper": "Evidence.pdf",
            "page": 1,
            "text": "The evidence supports the answer.",
            "score": 1.0,
        },
    ]
    state["verification"] = {
        "status": "complete",
        "covered": {"R1": ["E1"]},
        "uncertain": {},
        "missing": [],
        "corrective_queries": [],
    }
    llm = StubLLM({}, "The evidence supports the answer [E1].")

    answer = writer_node(state, llm)["answer"]  # type: ignore[arg-type]

    assert answer == "The evidence supports the answer [Evidence.pdf p.1]."
    assert "Answer in English" in llm.last_prompt
    assert "Answer in French" not in llm.last_prompt
    assert "Coverage status: complete" in llm.last_prompt


def test_writer_repairs_an_unverified_citation_once(sample_chunks: list[dict]) -> None:
    state = initial_state("Explain Self-RAG")
    state["plan"]["requirements"] = [
        requirement("R1", "Explain Self-RAG's mechanism", ["Self-RAG"]),
    ]
    state["evidence"] = sample_chunks[:2]
    state["verification"] = {
        "status": "complete",
        "covered": {"R1": ["E1"]},
        "uncertain": {},
        "missing": [],
        "corrective_queries": [],
    }

    repaired = writer_node(
        state,
        SequenceLLM(["Unsupported [E99].", "Supported [E1]."]),  # type: ignore[arg-type]
    )["answer"]
    assert repaired == "Supported [Self-RAG.pdf p.1]."


def test_writer_safely_abstains_when_citation_repair_fails(
    sample_chunks: list[dict],
) -> None:
    state = initial_state("Explain Self-RAG")
    state["plan"]["requirements"] = [
        requirement("R1", "Explain Self-RAG's mechanism", ["Self-RAG"]),
    ]
    state["evidence"] = sample_chunks[:2]
    state["verification"] = {
        "status": "complete",
        "covered": {"R1": ["E1"]},
        "uncertain": {},
        "missing": [],
        "corrective_queries": [],
    }

    failed_repair = writer_node(
        state,
        StubLLM({}, "Unsupported [E99]."),  # type: ignore[arg-type]
    )["answer"]
    assert failed_repair == SAFE_ABSTENTION

    state["verification"] = {
        "status": "insufficient",
        "covered": {},
        "uncertain": {},
        "missing": ["R1"],
        "corrective_queries": [],
    }
    advisory_answer = writer_node(
        state,
        StubLLM({}, "Unsupported [E1]."),  # type: ignore[arg-type]
    )["answer"]
    assert advisory_answer == "Unsupported [Self-RAG.pdf p.1]."

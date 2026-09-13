from __future__ import annotations

from typing import Any

import pytest

import scholar_agent.agents.researcher as researcher_module
from scholar_agent.agents.planner import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MIN_TOP_K,
    evidence_matches_target,
    planner_node,
    target_matches,
)
from scholar_agent.agents.researcher import (
    _attach_requirement_scores,
    _build_evidence_board,
    _select_candidates_for_reranking,
    _select_evidence,
    recovery_trace_entry,
    researcher_node,
)
from scholar_agent.agents.writer import (
    SAFE_ABSTENTION,
    citation_validator_node,
    writer_node,
)
from scholar_agent.config import Settings
from scholar_agent.workflow import initial_state


class StubLLM:
    def __init__(self, payload: object = None, text: str = "") -> None:
        self.payload = payload
        self.text = text
        self.last_prompt = ""
        self.complete_calls = 0

    def complete_json(self, prompt: str) -> dict[str, Any]:
        self.last_prompt = prompt
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload  # type: ignore[return-value]

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        self.complete_calls += 1
        return self.text


class FakeEngine:
    def __init__(
        self,
        sparse_by_query: dict[str, list[dict]],
        dense_by_query: dict[str, list[dict]],
    ) -> None:
        self.sparse_by_query = sparse_by_query
        self.dense_by_query = dense_by_query
        self.sparse_calls: list[tuple[list[str], int]] = []
        self.dense_calls: list[tuple[list[str], int]] = []

    def sparse_search(self, queries: list[str], top_k: int = 8) -> list[dict]:
        self.sparse_calls.append((queries, top_k))
        return self.sparse_by_query.get(queries[0], [])[:top_k]

    def dense_search_many(
        self,
        queries: list[str],
        top_k: int = 8,
    ) -> list[list[dict]]:
        self.dense_calls.append((queries, top_k))
        return [self.dense_by_query.get(query, [])[:top_k] for query in queries]


def requirement(
    requirement_id: str,
    description: str,
    targets: list[str],
    query: str,
    strategy: str = "hybrid",
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    return {
        "id": requirement_id,
        "description": description,
        "targets": targets,
        "query": query,
        "retrieval_strategy": strategy,
        "top_k": top_k,
    }


def scored(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
    return [
        {**item, "score": 1.0, "_query_scores": [1.0] * len(queries)}
        for item in candidates
    ]


def test_planner_accepts_all_supported_retrieval_strategies() -> None:
    state = initial_state("Compare Alpha, Beta, and Gamma")
    llm = StubLLM(
        {
            "requirements": [
                {
                    "description": "Explain Alpha",
                    "targets": ["Alpha"],
                    "query": "Alpha exact method",
                    "retrieval_strategy": "bm25",
                    "top_k": 4,
                },
                {
                    "description": "Explain Beta conceptually",
                    "targets": ["Beta"],
                    "query": "semantic mechanism of Beta",
                    "retrieval_strategy": "dense",
                    "top_k": 8,
                },
                {
                    "description": "Compare Alpha and Gamma",
                    "targets": ["Alpha", "Gamma"],
                    "query": "Alpha Gamma comparison",
                    "retrieval_strategy": "hybrid",
                    "top_k": 12,
                },
            ],
        },
    )

    plan = planner_node(state, llm)["plan"]  # type: ignore[arg-type]

    assert [item["retrieval_strategy"] for item in plan["requirements"]] == [
        "bm25",
        "dense",
        "hybrid",
    ]
    assert [item["top_k"] for item in plan["requirements"]] == [4, 8, 12]
    assert set(plan["requirements"][0]) == {
        "id",
        "description",
        "targets",
        "query",
        "retrieval_strategy",
        "top_k",
    }
    assert "Exact paper titles, acronyms" in llm.last_prompt
    assert "Conceptual mechanisms" in llm.last_prompt
    assert "Ambiguous comparisons" in llm.last_prompt
    assert "potentially ambiguous named entity" in llm.last_prompt
    assert "domain-relevant meaning or meanings" in llm.last_prompt
    assert "neutral entity-category terms" in llm.last_prompt
    assert "surface competing identities" in llm.last_prompt
    assert "what's CRAG" not in llm.last_prompt
    assert "do not predict the answer" in llm.last_prompt


def test_planner_defaults_factual_questions_to_research() -> None:
    llm = StubLLM(
        {
            "requirements": [
                {
                    "description": "Define agentic RAG briefly",
                    "targets": ["agentic rag"],
                    "query": "agentic RAG",
                    "retrieval_strategy": "bm25",
                    "top_k": 12,
                },
            ],
        },
    )

    result = planner_node(initial_state("do u know agentic rag"), llm)  # type: ignore[arg-type]
    plan = result["plan"]

    assert result["route"] == "research"
    assert plan == {
        "requirements": [
            requirement(
                "R1",
                "Define agentic RAG briefly",
                ["agentic rag"],
                "agentic RAG",
                "bm25",
                12,
            ),
        ],
    }
    assert "minimum requirements needed" in llm.last_prompt
    assert "should normally remain one" in llm.last_prompt


def test_planner_routes_non_factual_conversation_without_requirements() -> None:
    llm = StubLLM(
        {
            "route": "conversation",
            "direct_response": "Hello! What would you like to research?",
            "requirements": [],
        },
    )

    result = planner_node(initial_state("hello"), llm)  # type: ignore[arg-type]

    assert result == {
        "route": "conversation",
        "plan": {"requirements": []},
        "answer": "Hello! What would you like to research?",
    }
    assert "only when no evidence-grounded factual answer is requested" in llm.last_prompt
    assert "Do not use this route to answer\n  definitions, factual questions" in llm.last_prompt


def test_planner_keeps_a_compound_question_plan() -> None:
    llm = StubLLM(
        {
            "requirements": [
                {
                    "description": "Compare Alpha and Beta",
                    "targets": ["Alpha", "Beta"],
                    "query": "Alpha Beta comparison",
                    "retrieval_strategy": "hybrid",
                    "top_k": 8,
                },
            ],
        },
    )

    planner_node(
        initial_state("What is Alpha, and how does it compare with Beta?"),
        llm,  # type: ignore[arg-type]
    )["plan"]

    assert "User question" in llm.last_prompt


@pytest.mark.parametrize("strategy", [None, "sparse", "", 42])
def test_planner_invalid_or_missing_strategy_falls_back_to_hybrid(
    strategy: object,
) -> None:
    payload = {
        "requirements": [
            {
                "description": "Explain Alpha",
                "targets": ["Alpha"],
                "query": "Alpha",
                "top_k": 8,
            },
        ],
    }
    if strategy is not None:
        payload["requirements"][0]["retrieval_strategy"] = strategy

    plan = planner_node(
        initial_state("Explain Alpha"),
        StubLLM(payload),  # type: ignore[arg-type]
    )["plan"]

    assert plan["requirements"][0]["retrieval_strategy"] == "hybrid"


@pytest.mark.parametrize(
    ("raw_top_k", "expected"),
    [
        (MIN_TOP_K, MIN_TOP_K),
        (MAX_TOP_K, MAX_TOP_K),
        (1, MIN_TOP_K),
        (100, MAX_TOP_K),
        (8.5, DEFAULT_TOP_K),
        ("9", DEFAULT_TOP_K),
        (True, DEFAULT_TOP_K),
        (None, DEFAULT_TOP_K),
    ],
)
def test_planner_clamps_or_falls_back_for_invalid_top_k(
    raw_top_k: object,
    expected: int,
) -> None:
    plan = planner_node(
        initial_state("Explain Alpha"),
        StubLLM(
            {
                "requirements": [
                    {
                        "description": "Explain Alpha",
                        "targets": ["Alpha"],
                        "query": "Alpha",
                        "retrieval_strategy": "bm25",
                        "top_k": raw_top_k,
                    },
                ],
            },
        ),  # type: ignore[arg-type]
    )["plan"]

    assert plan["requirements"][0]["top_k"] == expected


def test_planner_repairs_targets_without_discarding_valid_requirements() -> None:
    plan = planner_node(
        initial_state("Explain Alpha and Beta"),
        StubLLM(
            {
                "requirements": [
                    {
                        "description": "Explain Alpha",
                        "targets": ["Alpha"],
                        "query": "",
                        "retrieval_strategy": "dense",
                        "top_k": 9,
                    },
                    {
                        "description": "Explain Alpha",
                        "targets": ["Alpha"],
                        "query": "duplicate",
                        "retrieval_strategy": "bm25",
                        "top_k": 4,
                    },
                    {"description": "", "targets": [], "query": "bad"},
                    "not an object",
                    {
                        "description": "Malformed targets",
                        "targets": [42],
                        "query": "bad",
                        "reason": "harmless extra field",
                    },
                    {
                        "description": "Invent Gamma",
                        "targets": ["Gamma"],
                        "query": "Gamma",
                    },
                ],
            },
        ),  # type: ignore[arg-type]
    )["plan"]

    assert plan["requirements"] == [
        requirement("R1", "Explain Alpha", ["Alpha"], "Explain Alpha", "dense", 9),
        requirement("R2", "Malformed targets", [], "bad", "hybrid", DEFAULT_TOP_K),
        requirement("R3", "Invent Gamma", [], "Gamma", "hybrid", DEFAULT_TOP_K),
    ]


def test_planner_ignores_extra_payload_fields_instead_of_falling_back() -> None:
    plan = planner_node(
        initial_state("Define Alpha"),
        StubLLM(
            {
                "reason": "The user asks for a short definition.",
                "requirements": [
                    {
                        "description": "Define Alpha",
                        "targets": ["Alpha", "Unstated target", None],
                        "query": "Alpha",
                        "retrieval_strategy": "dense",
                        "top_k": 8,
                        "rationale": "A semantic definition query.",
                    },
                ],
            },
        ),  # type: ignore[arg-type]
    )["plan"]

    assert plan["requirements"] == [
        requirement("R1", "Define Alpha", ["Alpha"], "Alpha", "dense", 8),
    ]


@pytest.mark.parametrize(
    "payload",
    [ValueError("bad JSON"), {}, {"requirements": []}, {"requirements": [None]}],
)
def test_planner_json_errors_fall_back_to_a_safe_requirement(payload: object) -> None:
    question = "Explain Alpha"

    plan = planner_node(
        initial_state(question),
        StubLLM(payload),  # type: ignore[arg-type]
    )["plan"]

    assert plan["requirements"] == [
        requirement("R1", question, [], question, "hybrid", DEFAULT_TOP_K),
    ]


def test_target_matching_preserves_method_identity() -> None:
    assert target_matches("Self-RAG", "Self RAG uses reflection tokens.")
    assert not target_matches("RAG", "CRAG and Self-RAG are methods.")
    assert evidence_matches_target(
        "Self-RAG",
        {"paper": "Self-RAG.pdf", "text": "The method retrieves passages."},
    )


def _state_with(requirements: list[dict]) -> dict:
    state = initial_state("Compare Self-RAG and CRAG")
    state["plan"] = {"requirements": requirements}
    return state


def test_bm25_requirement_does_not_call_dense(sample_chunks: list[dict]) -> None:
    request = requirement("R1", "Explain Self-RAG", ["Self-RAG"], "self", "bm25", 7)
    engine = FakeEngine({"self": sample_chunks[:1]}, {"self": sample_chunks[1:]})

    result = researcher_node(
        _state_with([request]),  # type: ignore[arg-type]
        engine,  # type: ignore[arg-type]
        Settings(),
        scored,
    )

    assert engine.sparse_calls == [(["self"], 7)]
    assert engine.dense_calls == []
    assert result["retrieval_trace"][0]["retrieval_strategy"] == "bm25"


def test_dense_requirement_does_not_call_bm25(sample_chunks: list[dict]) -> None:
    request = requirement("R1", "Explain Self-RAG", ["Self-RAG"], "self", "dense", 6)
    engine = FakeEngine({"self": sample_chunks[1:]}, {"self": sample_chunks[:1]})

    result = researcher_node(
        _state_with([request]),  # type: ignore[arg-type]
        engine,  # type: ignore[arg-type]
        Settings(),
        scored,
    )

    assert engine.sparse_calls == []
    assert engine.dense_calls == [(["self"], 6)]
    assert result["retrieval_trace"][0]["retrieval_strategy"] == "dense"


def test_hybrid_requirement_runs_both_and_rrf(
    sample_chunks: list[dict],
    monkeypatch: Any,
) -> None:
    request = requirement("R1", "Explain Self-RAG", ["Self-RAG"], "self", "hybrid", 10)
    engine = FakeEngine({"self": sample_chunks[:1]}, {"self": sample_chunks[1:2]})
    fusion_inputs: list[list[list[str]]] = []
    real_fusion = researcher_module.reciprocal_rank_fusion

    def fuse(*rankings: list[dict]) -> list[dict]:
        fusion_inputs.append(
            [[item["chunk_id"] for item in ranking] for ranking in rankings],
        )
        return real_fusion(*rankings)

    monkeypatch.setattr(researcher_module, "reciprocal_rank_fusion", fuse)
    result = researcher_node(
        _state_with([request]),  # type: ignore[arg-type]
        engine,  # type: ignore[arg-type]
        Settings(),
        scored,
    )

    assert engine.sparse_calls == [(["self"], 10)]
    assert engine.dense_calls == [(["self"], 10)]
    assert [["self-1"], ["crag-1"]] in fusion_inputs
    assert {item["chunk_id"] for item in result["evidence"]} == {"self-1", "crag-1"}


def test_one_question_can_mix_all_retrieval_strategies(sample_chunks: list[dict]) -> None:
    requirements = [
        requirement("R1", "Explain Self-RAG", ["Self-RAG"], "q1", "bm25", 4),
        requirement("R2", "Explain CRAG", ["CRAG"], "q2", "dense", 8),
        requirement("R3", "Compare both", [], "q3", "hybrid", 8),
    ]
    engine = FakeEngine(
        {"q1": sample_chunks[:1], "q3": sample_chunks[:2]},
        {"q2": sample_chunks[1:2], "q3": sample_chunks[:2]},
    )

    result = researcher_node(
        _state_with(requirements),  # type: ignore[arg-type]
        engine,  # type: ignore[arg-type]
        Settings(),
        scored,
    )

    assert engine.sparse_calls == [(["q1"], 4), (["q3"], 8)]
    assert engine.dense_calls == [(["q2", "q3"], 8)]
    assert result["retrieval_trace"] == [
        {
            "requirement_id": "R1",
            "query": "q1",
            "retrieval_strategy": "bm25",
            "top_k": 4,
        },
        {
            "requirement_id": "R2",
            "query": "q2",
            "retrieval_strategy": "dense",
            "top_k": 8,
        },
        {
            "requirement_id": "R3",
            "query": "q3",
            "retrieval_strategy": "hybrid",
            "top_k": 8,
        },
    ]


def test_recovery_trace_keeps_candidate_and_rerank_provenance(
    sample_chunks: list[dict],
) -> None:
    reranked = [
        {**sample_chunks[1], "score": 4.5},
        {**sample_chunks[0], "score": 2.0},
    ]

    trace = recovery_trace_entry(
        "R1",
        "search_within_paper",
        "manual_failure_validation",
        1,
        {"paper": "CRAG.pdf", "query": "correction", "top_k": 4},
        sample_chunks[:2],
        reranked,
    )

    assert trace == {
        "round": 1,
        "trigger": "manual_failure_validation",
        "requirement_id": "R1",
        "action": "search_within_paper",
        "parameters": {"paper": "CRAG.pdf", "query": "correction", "top_k": 4},
        "results": [
            {
                "chunk_id": "self-1",
                "paper": "Self-RAG.pdf",
                "page": 1,
                "candidate_rank": 1,
                "rerank_rank": 2,
                "rerank_score": 2.0,
            },
            {
                "chunk_id": "crag-1",
                "paper": "CRAG.pdf",
                "page": 2,
                "candidate_rank": 2,
                "rerank_rank": 1,
                "rerank_score": 4.5,
            },
        ],
    }


def test_requirement_aware_evidence_selection_still_reserves_each_requirement() -> None:
    requirements = [
        requirement("R1", "Explain mechanism", [], "mechanism"),
        requirement("R2", "Explain limitations", [], "limitations"),
    ]
    items = [
        {
            "chunk_id": f"mechanism-{index}",
            "paper": f"Mechanism-{index}.pdf",
            "page": 1,
            "text": "Mechanism evidence.",
            "score": 1.0 - index / 100,
            "_requirement_scores": {"R1": 1.0, "R2": 0.1},
        }
        for index in range(8)
    ]
    limitation = {
        "chunk_id": "limitation",
        "paper": "Limitations.pdf",
        "page": 1,
        "text": "Limitation evidence.",
        "score": 0.9,
        "_requirement_scores": {"R1": 0.1, "R2": 0.9},
    }

    selected = _select_evidence(items + [limitation], requirements, min_score=0.5)

    assert len(selected) == 8
    assert "limitation" in {item["chunk_id"] for item in selected}


def test_target_aware_allocation_and_page_diversity_still_work(
    sample_chunks: list[dict],
) -> None:
    duplicate_page = {
        **sample_chunks[0],
        "chunk_id": "self-duplicate",
        "score": 0.8,
        "_requirement_scores": {"R1": 0.8},
    }
    self_item = {
        **sample_chunks[0],
        "score": 0.9,
        "_requirement_scores": {"R1": 0.9},
    }
    crag_item = {
        **sample_chunks[1],
        "score": 0.7,
        "_requirement_scores": {"R1": 0.7},
    }
    requirements = [
        requirement("R1", "Compare Self-RAG and CRAG", ["Self-RAG", "CRAG"], "compare"),
    ]

    selected = _select_evidence(
        [self_item, duplicate_page, crag_item],
        requirements,
        min_score=0.5,
    )

    assert {item["chunk_id"] for item in selected} == {"self-1", "crag-1"}


def test_reranker_scores_remain_requirement_specific() -> None:
    result = _attach_requirement_scores(
        [
            {
                "chunk_id": "c1",
                "score": 0.9,
                "_query_scores": [0.9, 0.2],
            },
        ],
        [["R1"], ["R2"]],
    )

    assert result[0]["_requirement_scores"] == {"R1": 0.9, "R2": 0.2}
    assert "_query_scores" not in result[0]

    with pytest.raises(ValueError, match="one query score per query"):
        _attach_requirement_scores([{"chunk_id": "c1", "score": 1.0}], [["R1"]])


def test_evidence_board_preserves_shared_matches_gaps_and_unassigned_evidence(
    sample_chunks: list[dict],
) -> None:
    requirements = [
        requirement("R1", "Explain Self-RAG", ["Self-RAG"], "self"),
        requirement("R2", "Explain limitations", [], "limitations"),
        requirement("R3", "Explain CRAG", ["CRAG"], "crag"),
        requirement("R4", "Report unavailable results", [], "missing"),
    ]
    items = [
        {**sample_chunks[0], "title": "Self-RAG: Learning to Retrieve", "section": "2. Method",
         "_requirement_scores": {"R1": 0.8, "R2": 0.5, "R3": 0.1, "R4": -2.0}},
        {**sample_chunks[1], "_requirement_scores": {"R1": 0.1, "R2": 0.49, "R3": 0.7, "R4": -2.0}},
        {**sample_chunks[2], "_requirement_scores": {"R1": 0.1, "R2": 0.2, "R3": 0.1, "R4": -2.0}},
    ]

    evidence, board = _build_evidence_board(items, requirements, min_score=0.5)

    assert board == {
        "R1": {
            "requirement": "Explain Self-RAG", "evidence_ids": ["E1"],
            "candidate_papers": [], "status": "unknown", "covered": [], "missing": [],
            "action": None,
        },
        "R2": {
            "requirement": "Explain limitations", "evidence_ids": ["E1"],
            "candidate_papers": [], "status": "unknown", "covered": [], "missing": [],
            "action": None,
        },
        "R3": {
            "requirement": "Explain CRAG", "evidence_ids": ["E2"],
            "candidate_papers": [], "status": "unknown", "covered": [], "missing": [],
            "action": None,
        },
        "R4": {
            "requirement": "Report unavailable results", "evidence_ids": [],
            "candidate_papers": [], "status": "unknown", "covered": [], "missing": [],
            "action": None,
        },
    }
    assert [item["chunk_id"] for item in evidence] == [item["chunk_id"] for item in items]
    assert [item["id"] for item in evidence] == ["E1", "E2", "E3"]
    assert [item["supports"] for item in evidence] == [["R1", "R2"], ["R3"], []]
    assert evidence[0]["title"] == "Self-RAG: Learning to Retrieve"
    assert evidence[0]["section"] == "2. Method"
    assert evidence[1]["title"] is None
    assert evidence[1]["section"] is None
    assert evidence[0]["requirement_scores"] == items[0]["_requirement_scores"]
    assert all("_requirement_scores" not in item for item in evidence)
    assert all("id" not in item for item in items)


def test_evidence_board_links_relevant_aliases_without_literal_target_matching(
    sample_chunks: list[dict],
) -> None:
    requirements = [
        requirement("R1", "Explain Corrective RAG", ["Corrective RAG"], "corrective retrieval"),
    ]
    item = {**sample_chunks[1], "_requirement_scores": {"R1": 5.8}}
    assert not evidence_matches_target("Corrective RAG", item)

    evidence, board = _build_evidence_board([item], requirements, min_score=-1.0)

    assert board["R1"]["evidence_ids"] == ["E1"]
    assert evidence[0]["supports"] == ["R1"]


def test_board_orders_shared_evidence_by_each_requirement_score(sample_chunks: list[dict]) -> None:
    requirements = [
        requirement("R1", "First aspect", [], "first"),
        requirement("R2", "Second aspect", [], "second"),
    ]
    items = [
        {**sample_chunks[0], "_requirement_scores": {"R1": 2.0, "R2": 1.0}},
        {**sample_chunks[1], "_requirement_scores": {"R1": 1.0, "R2": 2.0}},
    ]

    evidence, board = _build_evidence_board(items, requirements, min_score=-1.0)

    assert board["R1"]["evidence_ids"] == ["E1", "E2"]
    assert board["R2"]["evidence_ids"] == ["E2", "E1"]
    assert [item["chunk_id"] for item in evidence] == ["self-1", "crag-1"]


def test_writer_uses_board_without_renumbering_or_hiding_selected_evidence(
    sample_chunks: list[dict],
) -> None:
    requirements = [
        requirement("R1", "Explain CRAG", ["CRAG"], "crag"),
        requirement("R2", "Explain Self-RAG", ["Self-RAG"], "self"),
        requirement("R3", "Report unavailable results", [], "missing"),
    ]
    items = [
        {**sample_chunks[0], "title": "Self-RAG: Learning to Retrieve", "section": "2. Method",
         "_requirement_scores": {"R1": -3.0, "R2": 2.0, "R3": -3.0}},
        {**sample_chunks[1], "_requirement_scores": {"R1": 2.0, "R2": -3.0, "R3": -3.0}},
        {**sample_chunks[2], "_requirement_scores": {"R1": -3.0, "R2": -3.0, "R3": -3.0}},
    ]
    state = _state_with(requirements)
    state["evidence"], state["evidence_board"] = _build_evidence_board(items, requirements, -1.0)
    state["evidence_board"]["R1"].update(
        status="sufficient",
        covered=["CRAG correction mechanism"],
    )
    state["evidence_board"]["R2"].update(
        status="sufficient",
        covered=["Self-RAG reflection mechanism"],
    )
    state["evidence_board"]["R3"].update(
        status="unresolved",
        missing=["requested results"],
    )
    llm = StubLLM(text="CRAG corrects retrieval [E2]. Self-RAG reflects [E1].")

    state.update(writer_node(state, llm))  # type: ignore[arg-type]
    answer = citation_validator_node(state)["answer"]

    assert answer == "CRAG corrects retrieval [CRAG.pdf p.2]. Self-RAG reflects [Self-RAG.pdf p.1]."
    assert llm.complete_calls == 1
    prompt = llm.last_prompt
    assert (
        "Requirement R1:\nExplain CRAG\nController coverage assessment:\n"
        "Status: sufficient\nCovered: CRAG correction mechanism\nMissing: None\n\n"
        "Candidate supporting evidence:\n[E2] CRAG.pdf — p.2"
    ) in prompt
    assert "[E1] Self-RAG: Learning to Retrieve (Self-RAG.pdf) — p.1 — 2. Method" in prompt
    assert (
        "Requirement R3:\nReport unavailable results\nController coverage assessment:\n"
        "Status: unresolved\nCovered: None identified\nMissing: requested results\n\n"
        "Candidate supporting evidence:\nNo matching evidence"
    ) in prompt
    assert "Additional candidate evidence (not linked to a requirement):\n[E3] Other.pdf — p.3" in prompt
    assert all(item["text"] in prompt for item in items)
    assert "Requirements are research scaffolding, not an answer outline" in prompt
    assert "Use only the necessary subset of the candidate evidence" in prompt
    assert "evidence availability alone is not" in prompt
    assert "Stop when the request is\nanswered" in prompt
    assert "Do not select one identity merely because its passage has the highest score" in prompt
    assert "Use a supplied Controller assessment instead of repeating its coverage analysis" in prompt
    assert "Mention a gap only when it blocks an important" in prompt
    assert "explicitly state which remaining requirements lack" not in prompt
    assert "Inspect all evidence yourself" not in prompt
    assert "retrieval_strategy" not in prompt


def test_candidate_pool_reserves_local_results_before_global_fill() -> None:
    rankings = [
        [
            {
                "chunk_id": f"q{query_index}-{rank}",
                "paper": "A.pdf",
                "page": rank + 1,
                "text": "evidence",
                "score": 0.0,
            }
            for rank in range(8)
        ]
        for query_index in range(5)
    ]

    candidates = _select_candidates_for_reranking(rankings)

    assert len(candidates) == 30
    for query_index in range(5):
        assert f"q{query_index}-0" in {item["chunk_id"] for item in candidates}


@pytest.mark.parametrize("filter_scores", [True, False])
def test_researcher_records_pages_before_candidate_and_evidence_filtering(
    filter_scores: bool,
) -> None:
    chunks = [
        {
            "chunk_id": str(index),
            "paper": "A.pdf",
            "page": index + 1,
            "text": "evidence",
            "score": 0.0,
        }
        for index in range(40)
    ]
    requests = [
        requirement("R1", "Explain A", [], "q1", "hybrid", 10),
        requirement("R2", "Explain B", [], "q2", "hybrid", 10),
    ]
    # Two chunks on the same page must not inflate page recall.
    sparse = [chunks[0], {**chunks[0], "chunk_id": "duplicate"}, *chunks[1:19]]
    engine = FakeEngine(
        {"q1": sparse[:10], "q2": sparse[10:]},
        {"q1": chunks[20:30], "q2": chunks[30:]},
    )
    observed_candidates = []

    def rerank(queries: list[str], candidates: list[dict], model: str) -> list[dict]:
        observed_candidates.extend(candidates)
        return [
            {
                **item,
                "score": 1.0 if index == 0 or not filter_scores else -10.0,
                "_query_scores": [1.0 if index == 0 or not filter_scores else -10.0] * len(queries),
            }
            for index, item in enumerate(candidates)
        ]

    result = researcher_node(
        _state_with(requests),  # type: ignore[arg-type]
        engine,  # type: ignore[arg-type]
        Settings(),
        rerank,
    )

    stages = result["retrieval_stages"]
    assert len(observed_candidates) == 30
    assert {item["page"] for item in stages["retrieval"]} == set(range(1, 20)) | set(range(21, 41))
    assert len(stages["retrieval"]) == 39
    assert stages["rerank"] == [
        {"paper": "A.pdf", "page": page}
        for page in sorted({item["page"] for item in observed_candidates})
    ]
    assert len(stages["rerank"]) < len(stages["retrieval"])
    assert len(result["evidence"]) == (1 if filter_scores else 4)
    assert len(stages["rerank"]) > len(result["evidence"])


def test_writer_and_deterministic_citation_validation_do_not_regress(
    sample_chunks: list[dict],
) -> None:
    state = _state_with(
        [requirement("R1", "Explain Self-RAG", ["Self-RAG"], "Self-RAG", "bm25")],
    )
    state["evidence"], state["evidence_board"] = _build_evidence_board(
        [{**sample_chunks[0], "_requirement_scores": {"R1": 1.0}}],
        state["plan"]["requirements"],
        min_score=-1.0,
    )
    llm = StubLLM(
        text="Supported [E1]. Unknown [E99]. Fabricated [Fake.pdf p.999].",
    )

    draft = writer_node(state, llm)  # type: ignore[arg-type]
    state.update(draft)
    validated = citation_validator_node(state)  # type: ignore[arg-type]

    assert draft["answer"] == "Supported [E1]. Unknown [E99]. Fabricated [Fake.pdf p.999]."
    assert validated["answer"] == "Supported [Self-RAG.pdf p.1]. Unknown. Fabricated."
    assert "Answer in English" in llm.last_prompt
    assert "Requirement–Evidence Blackboard:" in llm.last_prompt
    assert "infer the smallest set of claims needed" in llm.last_prompt
    assert "evidence availability alone is not" in llm.last_prompt
    assert "A citation supports only the sentence in which it appears" in llm.last_prompt


def test_writer_abstains_without_evidence_or_an_llm_call() -> None:
    state = initial_state("Missing evidence")
    llm = StubLLM(text="must not be used")

    draft = writer_node(state, llm)  # type: ignore[arg-type]
    state.update(draft)

    assert draft["answer"] == SAFE_ABSTENTION
    assert citation_validator_node(state)["answer"] == SAFE_ABSTENTION
    assert llm.complete_calls == 0

from __future__ import annotations

from scholar_agent.agents.controller import controller_node, sanitize_actions
from scholar_agent.workflow import initial_state


class StubLLM:
    def __init__(self, payload: dict):
        self.payload = payload
        self.prompt = ""

    def complete_json(self, prompt: str) -> dict:
        self.prompt = prompt
        return self.payload


def _controller_state(sample_chunks: list[dict]) -> dict:
    state = initial_state("Compare Self-RAG and CRAG")
    state["plan"] = {"requirements": [
        {
            "id": "R1", "description": "Explain Self-RAG", "targets": ["Self-RAG"],
            "query": "Self-RAG mechanism", "retrieval_strategy": "dense", "top_k": 8,
        },
        {
            "id": "R2", "description": "Explain CRAG", "targets": ["CRAG"],
            "query": "CRAG mechanism", "retrieval_strategy": "hybrid", "top_k": 12,
        },
    ]}
    state["evidence"] = [
        {
            **item, "id": f"E{index}", "paper_id": item["paper"],
            "supports": [f"R{index}"], "requirement_scores": {f"R{index}": 2.0},
        }
        for index, item in enumerate(sample_chunks[:2], start=1)
    ]
    state["evidence_board"] = {
        "R1": {
            "requirement": "Explain Self-RAG", "evidence_ids": ["E1"],
            "candidate_papers": [{
                "paper": "Self-RAG.pdf", "title": "Self-RAG",
                "best_score": 2.0, "selected": True,
            }],
        },
        "R2": {
            "requirement": "Explain CRAG", "evidence_ids": ["E2"],
            "candidate_papers": [{
                "paper": "CRAG.pdf", "title": "CRAG",
                "best_score": 2.0, "selected": True,
            }],
        },
    }
    return state


def test_controller_keeps_two_bounded_actions_for_distinct_requirements(sample_chunks):
    state = _controller_state(sample_chunks)
    payload = {"actions": [
        {
            "requirement_id": "R1", "action": "search_within_paper",
            "candidate_id": "P1", "query": "reflection tokens", "reason": "detail missing",
        },
        {
            "requirement_id": "R1", "action": "expand_neighbors",
            "chunk_id": "self-1", "query": "duplicate requirement",
        },
        {
            "requirement_id": "R2", "action": "expand_neighbors",
            "chunk_id": "crag-1", "query": "correction details",
        },
    ]}

    actions, rejections = sanitize_actions(payload, state)  # type: ignore[arg-type]

    assert [item["action"] for item in actions] == ["search_within_paper", "expand_neighbors"]
    assert [item["requirement_id"] for item in actions] == ["R1", "R2"]
    assert actions[0]["paper"] == "Self-RAG.pdf"
    assert rejections == [{
        "requirement_id": "R1",
        "action": "expand_neighbors",
        "reason": "duplicate_requirement",
        "chunk_id": "self-1",
    }]


def test_controller_rejection_retains_invalid_selector(sample_chunks):
    state = _controller_state(sample_chunks)

    actions, rejections = sanitize_actions({"actions": [{
        "requirement_id": "R1", "action": "search_within_paper",
        "candidate_id": "P9", "query": "reflection tokens",
    }]}, state)  # type: ignore[arg-type]

    assert actions == []
    assert rejections == [{
        "requirement_id": "R1", "action": "search_within_paper",
        "reason": "unknown_candidate", "candidate_id": "P9",
    }]


def test_controller_prompt_contains_observation_and_no_answer_labels(sample_chunks):
    state = _controller_state(sample_chunks)
    llm = StubLLM({"actions": []})

    result = controller_node(state, llm)  # type: ignore[arg-type]

    assert result == {
        "controller_trace": {"actions": [], "rejected_actions": 0, "rejections": []},
    }
    assert "Self-RAG uses adaptive retrieval" in llm.prompt
    assert "chunk_id=self-1" in llm.prompt
    assert "Observed candidate papers" in llm.prompt
    assert '[P1] Self-RAG' in llm.prompt
    assert '`"candidate_id": "P1"`' in llm.prompt
    assert "gold_pages" not in llm.prompt
    assert "answer_key" not in llm.prompt

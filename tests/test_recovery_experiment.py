from __future__ import annotations

from copy import deepcopy

from evals import evaluate_recovery as experiment

from scholar_agent.agents.researcher import _build_evidence_board
from scholar_agent.config import Settings


class PaperEngine:
    def __init__(self, candidates: list[dict]):
        self.candidates = candidates

    def search_within_paper(self, paper: str, query: str, top_k: int = 4) -> list[dict]:
        return self.candidates[:top_k]

    def expand_neighbors(self, chunk_id: str, radius: int = 1) -> list[dict]:
        return self.candidates


def test_paper_recovery_reranks_and_adds_at_most_two_passages(sample_chunks):
    plan = {"requirements": [{
        "id": "R1", "description": "Explain CRAG", "targets": ["CRAG"],
        "query": "CRAG mechanisms", "retrieval_strategy": "hybrid", "top_k": 8,
    }]}
    baseline, board = _build_evidence_board(
        [{**sample_chunks[0], "_requirement_scores": {"R1": 2.0}}], plan["requirements"], -1.0,
    )
    candidates = [
        {**deepcopy(sample_chunks[1]), "chunk_id": f"crag-{index}", "score": 0.0}
        for index in range(3)
    ]
    state = {
        "plan": plan,
        "evidence": baseline,
        "evidence_board": board,
        "retrieval_stages": {"retrieval": []},
    }
    spec = {
        "requirement_id": "R1", "critical_chunk_id": "crag-0",
        "action": "paper_navigation", "paper": "CRAG.pdf", "query": "decompose recompose",
    }

    def fake_rerank(queries, items, model):
        return [
            {**item, "score": 3.0 - index, "_query_scores": [3.0 - index]}
            for index, item in enumerate(items)
        ]

    updates, chain = experiment.recover_state(
        state, PaperEngine(candidates), Settings(min_rerank_score=-1.0), spec,
        rerank_function=fake_rerank,
    )

    assert [item["chunk_id"] for item in updates["evidence"]] == ["self-1", "crag-0", "crag-1"]
    assert chain == {
        "chunk_id": "crag-0", "recovered": True, "reranked": True,
        "passed_threshold": True, "selected": True,
    }
    assert [item["action"] for item in updates["recovery_trace"]] == [
        "search_within_paper", "expand_neighbors",
    ]

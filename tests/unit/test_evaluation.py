"""Phase 8 evaluation: dataset freeze, metrics, ablation with fakes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.evaluation.ablation import AblationConfig, evaluate_one, run_ablation
from scholar_agent.evaluation.answer_metrics import (
    compute_answer_metrics,
    is_refusal,
    token_f1,
)
from scholar_agent.evaluation.baselines import SystemRunner, _merge_hits
from scholar_agent.evaluation.citation_metrics import compute_citation_metrics_from_papers
from scholar_agent.evaluation.dataset import (
    EvalQuestion,
    GoldEvidence,
    compute_dataset_fingerprint,
    load_eval_dataset,
    validate_dataset_against_store,
)
from scholar_agent.evaluation.report import write_report
from scholar_agent.evaluation.retrieval_metrics import compute_retrieval_metrics
from scholar_agent.ids import make_chunk_id
from scholar_agent.models.retrieval import RetrievalHit
from scholar_agent.retrieval.chunk_store import ChunkStore
from scholar_agent.retrieval.tools import RetrievalToolkit

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "data" / "evaluation"


@pytest.fixture(scope="module")
def frozen_dataset():
    if not (EVAL_DIR / "questions.jsonl").is_file():
        pytest.skip("frozen evaluation dataset not present")
    return load_eval_dataset(
        questions_path=EVAL_DIR / "questions.jsonl",
        reference_evidence_path=EVAL_DIR / "reference_evidence.jsonl",
        frozen_split_path=EVAL_DIR / "frozen_split.json",
        validate=True,
    )


def test_frozen_dataset_has_50_questions_by_type(frozen_dataset) -> None:
    assert len(frozen_dataset.questions) == 50
    counts: dict[str, int] = {}
    for q in frozen_dataset.questions:
        counts[q.question_type] = counts.get(q.question_type, 0) + 1
    assert counts == {
        "factual": 10,
        "keyword": 10,
        "comparison": 15,
        "relational": 10,
        "unanswerable": 5,
    }
    assert frozen_dataset.split is not None
    assert frozen_dataset.split.frozen is True
    assert len(frozen_dataset.split.question_ids) == 50


def test_fingerprint_matches_frozen_split(frozen_dataset) -> None:
    fp = compute_dataset_fingerprint(
        EVAL_DIR / "questions.jsonl",
        EVAL_DIR / "reference_evidence.jsonl",
    )
    assert frozen_dataset.split is not None
    assert fp == frozen_dataset.split.fingerprint_sha256


def test_frozen_gold_maps_to_canonical_store(frozen_dataset) -> None:
    processed = REPO / "data" / "processed" / "chunks.jsonl"
    if not processed.is_file():
        pytest.skip("local processed chunk store not present (run ingest)")
    store = ChunkStore.from_processed_dir(REPO / "data" / "processed")
    assert validate_dataset_against_store(frozen_dataset, store) == []


def test_tampered_dataset_fails_validation(tmp_path: Path, frozen_dataset) -> None:
    q_path = tmp_path / "questions.jsonl"
    # Drop one question
    lines = (EVAL_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    q_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    r_path = tmp_path / "reference_evidence.jsonl"
    r_path.write_text(
        (EVAL_DIR / "reference_evidence.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    s_path = tmp_path / "frozen_split.json"
    s_path.write_text(
        (EVAL_DIR / "frozen_split.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid|fingerprint|50"):
        load_eval_dataset(
            questions_path=q_path,
            reference_evidence_path=r_path,
            frozen_split_path=s_path,
            validate=True,
        )


def test_retrieval_metrics_recall_and_mrr() -> None:
    q = EvalQuestion(
        question_id="t1",
        question="What is Self-RAG?",
        question_type="factual",
        required_paper_ids=["paper_a"],
        required_chunk_ids=["chunk_gold"],
        gold_evidence=[
            GoldEvidence(paper_id="paper_a", chunk_id="chunk_gold", page_start=1, page_end=1)
        ],
    )
    hits = [
        RetrievalHit(
            chunk_id="chunk_other",
            paper_id="paper_b",
            text="other",
            page_start=1,
            page_end=1,
            retrieval_method="dense",
        ),
        RetrievalHit(
            chunk_id="chunk_gold",
            paper_id="paper_a",
            text="self-rag",
            page_start=1,
            page_end=1,
            retrieval_method="dense",
        ),
    ]
    m = compute_retrieval_metrics(q, hits, k=5)
    assert m.recall_at_k == 1.0
    assert m.mrr == pytest.approx(0.5)
    assert m.hit_at_k == 1.0
    assert m.recall_at_k_paper == 1.0


def test_graph_evidence_recall_uses_only_graph_hits() -> None:
    question = EvalQuestion(
        question_id="graph_1",
        question="How are A and B related?",
        question_type="relational",
        required_paper_ids=["paper_a"],
        required_chunk_ids=["chunk_gold"],
        graph_reasoning_expected=True,
    )
    hits = [
        RetrievalHit(
            chunk_id="chunk_gold",
            paper_id="paper_a",
            text="relation",
            page_start=1,
            page_end=1,
            retrieval_method="graph",
        )
    ]
    metrics = compute_retrieval_metrics(question, hits, k=5)
    assert metrics.graph_evidence_recall == 1.0


def test_multi_tool_merge_does_not_starve_graph_hits() -> None:
    def hit(chunk_id: str, method: str) -> RetrievalHit:
        return RetrievalHit(
            chunk_id=chunk_id,
            paper_id=f"paper_{chunk_id}",
            text=chunk_id,
            page_start=1,
            page_end=1,
            retrieval_method=method,
        )

    hybrid = [hit(f"hybrid_{index}", "hybrid_rerank") for index in range(8)]
    graph = [hit(f"graph_{index}", "graph") for index in range(8)]
    merged = _merge_hits(hybrid, graph, limit=8)
    assert len(merged) == 8
    assert sum(item.retrieval_method == "graph" for item in merged) == 4


def test_citation_and_answer_metrics_unanswerable() -> None:
    q = EvalQuestion(
        question_id="u1",
        question="What is the stock price?",
        question_type="unanswerable",
        unanswerable=True,
        reference_claims=["Corpus cannot answer."],
    )
    cite = compute_citation_metrics_from_papers(q, set())
    assert cite.citation_precision == 1.0
    ans = compute_answer_metrics(
        q,
        "Limitation: the corpus does not contain live stock prices.",
        contexts=[],
    )
    assert ans.refusal_correct == 1.0
    assert is_refusal("Limitation: the corpus does not contain live stock prices.")


def test_token_f1_basic() -> None:
    assert token_f1("self rag reflection tokens", "self-rag uses reflection tokens") > 0.3
    assert token_f1("", "") == 1.0


def test_ragas_uses_only_explicit_evaluator() -> None:
    question = EvalQuestion(
        question_id="r1",
        question="What is RAG?",
        question_type="factual",
        reference_answer="RAG augments generation with retrieved evidence.",
        reference_claims=["RAG uses retrieved evidence."],
    )
    calls: list[dict[str, object]] = []

    def evaluator(**kwargs: object) -> dict[str, float]:
        calls.append(kwargs)
        return {"faithfulness": 0.75, "answer_relevancy": 0.8}

    without_provider = compute_answer_metrics(
        question,
        "RAG uses retrieved evidence.",
        contexts=["RAG augments generation with retrieval."],
        use_ragas=True,
    )
    assert without_provider.used_ragas is False

    with_provider = compute_answer_metrics(
        question,
        "RAG uses retrieved evidence.",
        contexts=["RAG augments generation with retrieval."],
        use_ragas=True,
        ragas_evaluator=evaluator,
    )
    assert len(calls) == 1
    assert with_provider.used_ragas is True
    assert with_provider.ragas_faithfulness == 0.75
    assert with_provider.ragas_answer_relevancy == 0.8


class _FakeToolkit(RetrievalToolkit):
    def __init__(self, hits_by_mode: dict[str, list[RetrievalHit]] | None = None) -> None:
        self.store = None  # type: ignore[assignment]
        self.dense = object()
        self.sparse = object()
        self.graph = object()
        self.reranker = None  # type: ignore[assignment]
        self.dense_top_k = 8
        self.sparse_top_k = 8
        self.fused_top_k = 8
        self.rerank_top_k = 8
        self.rrf_k = 60
        self.hits_by_mode = hits_by_mode or {}
        self.calls: list[str] = []

    def search(self, query: str, *, mode: str = "hybrid_rerank", k=None, filters=None):  # type: ignore[override]
        from scholar_agent.models.retrieval import RetrievalResult

        self.calls.append(mode)
        hits = list(self.hits_by_mode.get(mode) or self.hits_by_mode.get("*") or [])
        method = (
            mode if mode in {"dense", "sparse", "hybrid", "hybrid_rerank", "graph"} else "hybrid"
        )
        return RetrievalResult(query=query, method=method, hits=hits[: k or 8])  # type: ignore[arg-type]


def test_ablation_runs_all_selected_systems_same_questions(tmp_path: Path) -> None:
    text = "Self-RAG retrieves on demand using reflection tokens."
    gold_chunk = make_chunk_id("paper_arxiv_2310_11511", page_start=1, page_end=1, text=text)
    hit = RetrievalHit(
        chunk_id=gold_chunk,
        paper_id="paper_arxiv_2310_11511",
        text=text,
        page_start=1,
        page_end=1,
        score=0.9,
        retrieval_method="hybrid_rerank",
    )
    toolkit = _FakeToolkit(
        hits_by_mode={
            "*": [hit],
            "dense": [hit],
            "sparse": [hit],
            "hybrid": [hit],
            "hybrid_rerank": [hit],
            "graph": [hit],
        }
    )

    questions = [
        EvalQuestion(
            question_id="q_f_test",
            question="What is Self-RAG?",
            question_type="factual",
            reference_answer="Self-RAG retrieves on demand.",
            reference_claims=["Self-RAG retrieves on demand using reflection tokens."],
            required_paper_ids=["paper_arxiv_2310_11511"],
            required_chunk_ids=[gold_chunk],
            gold_evidence=[
                GoldEvidence(
                    paper_id="paper_arxiv_2310_11511",
                    chunk_id=gold_chunk,
                    page_start=1,
                    page_end=1,
                )
            ],
        ),
        EvalQuestion(
            question_id="q_u_test",
            question="What is the 2028 Turing Award winner for LightRAG?",
            question_type="unanswerable",
            unanswerable=True,
            reference_claims=["Not in corpus."],
        ),
    ]
    from scholar_agent.evaluation.dataset import EvalDataset

    dataset = EvalDataset(questions=questions)
    runner = SystemRunner(toolkit, top_k=5)  # type: ignore[arg-type]

    # Patch workflow systems to avoid long agent loops: only retrieval systems
    systems = ["naive_dense", "hybrid_rag", "hybrid_rerank", "static_all_tools"]
    report, rows = run_ablation(
        dataset,
        runner,
        AblationConfig(systems=systems, top_k=5, use_ragas=False),
    )
    assert len(rows) == len(systems) * 2
    assert {s.system for s in report.systems} == set(systems)
    # Identical question coverage
    for system in systems:
        ids = [r.question.question_id for r in rows if r.output.system == system]
        assert ids == ["q_f_test", "q_u_test"]
    assert report.config["question_ids"] == ["q_f_test", "q_u_test"]
    for summary in report.systems:
        assert "avg_latency_ms" in summary.by_type["factual"]
        assert "estimated_cost_usd" in summary.by_type["factual"]

    paths = write_report(report, tmp_path)
    assert paths["results_json"].is_file()
    assert paths["run_config_json"].is_file()
    assert paths["aggregate_csv"].is_file()
    assert paths["chart_recall"].is_file()
    assert paths["chart_category_factual"].is_file()
    payload = json.loads(paths["results_json"].read_text(encoding="utf-8"))
    assert payload["run_id"]
    assert len(payload["systems"]) == len(systems)


def test_evaluate_one_records_latency_and_cost() -> None:
    q = EvalQuestion(
        question_id="q1",
        question="What is DPR?",
        question_type="factual",
        required_paper_ids=["paper_x"],
        gold_evidence=[GoldEvidence(paper_id="paper_x", page_start=1, page_end=1)],
    )
    hit = RetrievalHit(
        chunk_id="c1",
        paper_id="paper_x",
        text="Dense Passage Retrieval dual encoder",
        page_start=1,
        page_end=1,
        retrieval_method="dense",
    )
    runner = SystemRunner(
        _FakeToolkit(hits_by_mode={"dense": [hit], "*": [hit]}),  # type: ignore[arg-type]
        top_k=3,
        usd_per_1k_tokens=0.01,
    )
    result = evaluate_one(runner, "naive_dense", q, top_k=3, use_ragas=False)
    assert result.output.latency_ms >= 0
    assert result.output.token_estimate > 0
    assert result.retrieval.recall_at_k_paper == 1.0
    # Fake toolkit has no canonical store, so validity is unknown/zero—not fabricated as 1.
    assert result.citation.citation_validity_rate == 0.0

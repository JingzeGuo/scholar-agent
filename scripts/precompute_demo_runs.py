#!/usr/bin/env python3
"""Precompute offline demo runs for interview-safe Streamlit replay.

Preference order:
1. Live run against local indexes (hash embedder) when available.
2. Fall back to curated offline fixtures so the demo works with no indexes.

Writes JSON under data/demo/runs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scholar_agent.app.demo_models import (
    DemoSessionResult,
    DemoSettings,
    NaiveComparisonView,
    SavedDemoRun,
    TraceSummary,
)
from scholar_agent.app.demo_runs import save_demo_run
from scholar_agent.app.demo_service import DemoService
from scholar_agent.app.status import collect_system_status
from scholar_agent.config import load_config
from scholar_agent.logging import setup_logging
from scholar_agent.models.answer import CitationReport, FinalAnswer, SourceCard
from scholar_agent.models.base import EventType, ExecutionEvent, QueryType, utc_now_iso
from scholar_agent.models.evidence import EvidenceItem
from scholar_agent.models.planning import QueryPlan, SubQuestion, SubQuestionStatus
from scholar_agent.models.workflow import VerificationResult

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "demo" / "runs"


def _fixture_compare() -> SavedDemoRun:
    run_id = "run_demo_selfrag_crag"
    settings = DemoSettings(
        compare_naive_rag=True,
        enable_graph=True,
        enable_corrective=True,
        static_routing=False,
        verified_evidence_only=True,
        embedding_backend="hash",
    )
    evidence = [
        EvidenceItem(
            evidence_id="ev_selfrag",
            sub_question_id="sq_0",
            claim="Self-RAG uses reflection tokens to retrieve on demand.",
            evidence_text=(
                "Self-RAG retrieves on demand and uses reflection tokens to critique "
                "generation quality."
            ),
            paper_id="paper_arxiv_2310_11511",
            chunk_id="chunk_selfrag_demo",
            page_start=1,
            page_end=2,
            retrieval_method="hybrid_rerank",
            retrieval_score=0.91,
        ),
        EvidenceItem(
            evidence_id="ev_crag",
            sub_question_id="sq_1",
            claim="CRAG evaluates retrieved documents and triggers corrective retrieval.",
            evidence_text=(
                "Corrective Retrieval Augmented Generation (CRAG) evaluates retrieved "
                "documents and triggers corrective retrieval when quality is low."
            ),
            paper_id="paper_arxiv_2401_15884",
            chunk_id="chunk_crag_demo",
            page_start=1,
            page_end=2,
            retrieval_method="hybrid_rerank",
            retrieval_score=0.88,
        ),
    ]
    cards = [
        SourceCard(
            evidence_id=e.evidence_id,
            paper_id=e.paper_id,
            chunk_id=e.chunk_id,
            page_start=e.page_start,
            page_end=e.page_end,
            snippet=e.evidence_text[:200],
            retrieval_method=e.retrieval_method,
            title="Self-RAG" if "2310" in e.paper_id else "CRAG",
            pdf_path=f"data/papers/{'2310.11511' if '2310' in e.paper_id else '2401.15884'}.pdf",
        )
        for e in evidence
    ]
    plan = QueryPlan(
        original_query="Compare Self-RAG versus CRAG",
        answer_type="comparison",
        expected_source_diversity=2,
        sub_questions=[
            SubQuestion(
                id="sq_0",
                question="What is Self-RAG?",
                query_type=QueryType.COMPARISON,
                required_evidence=["definition"],
                status=SubQuestionStatus.COVERED,
            ),
            SubQuestion(
                id="sq_1",
                question="What is CRAG?",
                query_type=QueryType.COMPARISON,
                required_evidence=["definition"],
                status=SubQuestionStatus.COVERED,
            ),
        ],
    )
    verification = VerificationResult(
        is_sufficient=True,
        coverage_score=1.0,
        covered_sub_questions=["sq_0", "sq_1"],
        rationale_summary="Both comparison sides covered after one corrective pass.",
        corrective_queries=["CRAG corrective retrieval evaluator"],
    )
    final = FinalAnswer(
        markdown=(
            "## Answer\n\n"
            "**Question:** Compare Self-RAG versus CRAG\n\n"
            "### Claims\n\n"
            "- Self-RAG uses reflection tokens to retrieve on demand "
            "[paper_arxiv_2310_11511 p.1-2]\n"
            "- CRAG evaluates retrieved documents and triggers corrective retrieval "
            "[paper_arxiv_2401_15884 p.1-2]\n"
        ),
        claims=[
            {
                "claim_id": "claim_1",
                "text": "Self-RAG uses reflection tokens to retrieve on demand.",
                "evidence_ids": ["ev_selfrag"],
            },
            {
                "claim_id": "claim_2",
                "text": "CRAG evaluates retrieved documents and triggers corrective retrieval.",
                "evidence_ids": ["ev_crag"],
            },
        ],
        sources=[c.format_reference() for c in cards],
        source_cards=cards,
        citation_report=CitationReport(
            is_valid=True,
            cited_evidence_ids=["ev_selfrag", "ev_crag"],
            cited_paper_ids=[
                "paper_arxiv_2310_11511",
                "paper_arxiv_2401_15884",
            ],
        ),
    )
    # claims need ClaimWithCitations - FinalAnswer expects list[ClaimWithCitations]
    from scholar_agent.models.answer import ClaimWithCitations

    final = final.model_copy(
        update={
            "claims": [
                ClaimWithCitations(
                    claim_id="claim_1",
                    text="Self-RAG uses reflection tokens to retrieve on demand.",
                    evidence_ids=["ev_selfrag"],
                ),
                ClaimWithCitations(
                    claim_id="claim_2",
                    text=(
                        "CRAG evaluates retrieved documents and triggers "
                        "corrective retrieval."
                    ),
                    evidence_ids=["ev_crag"],
                ),
            ]
        }
    )
    events = [
        ExecutionEvent(
            run_id=run_id,
            event_type=EventType.PLAN_CREATED,
            component="planner",
            summary="plan answer_type=comparison sub_questions=2",
        ),
        ExecutionEvent(
            run_id=run_id,
            event_type=EventType.TOOL_RESULT,
            component="researcher",
            summary="hybrid_rerank hits for Self-RAG",
            payload={"method": "hybrid_rerank", "tool_name": "hybrid_search"},
        ),
        ExecutionEvent(
            run_id=run_id,
            event_type=EventType.VERIFICATION,
            component="verifier",
            summary="missing CRAG side",
            payload={"is_sufficient": False},
        ),
        ExecutionEvent(
            run_id=run_id,
            event_type=EventType.CORRECTIVE,
            component="workflow",
            summary="corrective iteration 1: 1 queries",
        ),
        ExecutionEvent(
            run_id=run_id,
            event_type=EventType.CITATION_VALIDATED,
            component="citation_validator",
            summary="citation valid=True claims=2",
            payload={"is_valid": True},
        ),
        ExecutionEvent(
            run_id=run_id,
            event_type=EventType.RUN_FINISHED,
            component="workflow",
            summary="workflow finished: evidence_sufficient",
        ),
    ]
    session = DemoSessionResult(
        run_id=run_id,
        query="Compare Self-RAG versus CRAG",
        settings=settings,
        offline_replay=True,
        answer_markdown=final.markdown,
        claims=[c.model_dump(mode="json") for c in final.claims],
        source_cards=cards,
        evidence=evidence,
        plan=plan,
        verification=verification,
        final_answer=final,
        events=events,
        trace=TraceSummary(
            query_type="comparison",
            answer_type="comparison",
            sub_questions=[
                {
                    "id": "sq_0",
                    "question": "What is Self-RAG?",
                    "status": "covered",
                    "query_type": "comparison",
                },
                {
                    "id": "sq_1",
                    "question": "What is CRAG?",
                    "status": "covered",
                    "query_type": "comparison",
                },
            ],
            tool_events=[
                {
                    "event_type": "tool_result",
                    "component": "researcher",
                    "summary": "hybrid_rerank hits",
                    "payload": {"method": "hybrid_rerank"},
                }
            ],
            retrieval_methods=["hybrid_rerank"],
            evidence_count=2,
            verified_evidence_count=2,
            coverage_score=1.0,
            is_sufficient=True,
            corrective_iterations=1,
            corrective_queries=["CRAG corrective retrieval evaluator"],
            citation_valid=True,
            citation_issue_count=0,
            terminated_reason="evidence_sufficient",
            unanswerable=False,
            latency_ms=420,
            tool_call_count=4,
            token_estimate=900,
        ),
        naive=NaiveComparisonView(
            answer=(
                "Evidence notes (Naive RAG):\n"
                "- [paper_arxiv_2310_11511 p.1] Self-RAG retrieves on demand…\n"
            ),
            method="naive_rag:hybrid_rerank",
            hit_count=4,
            latency_ms=55,
            used_llm=False,
        ),
        status={"ok": True, "offline_fixture": True},
    )
    return SavedDemoRun(
        demo_id="selfrag_vs_crag",
        title="Self-RAG vs CRAG (corrective loop)",
        query=session.query,
        settings=settings,
        created_at=utc_now_iso(),
        offline=True,
        notes="Shows corrective iteration when one comparison side is initially missing.",
        session=session,
    )


def _fixture_factual() -> SavedDemoRun:
    run_id = "run_demo_what_is_selfrag"
    settings = DemoSettings(compare_naive_rag=True, enable_graph=False, enable_corrective=False)
    card = SourceCard(
        evidence_id="ev_def",
        paper_id="paper_arxiv_2310_11511",
        chunk_id="chunk_selfrag_def",
        page_start=1,
        page_end=1,
        snippet="Self-RAG learns to retrieve, generate, and critique through self-reflection.",
        retrieval_method="hybrid_rerank",
        title="Self-RAG",
        pdf_path="data/papers/2310.11511.pdf",
    )
    from scholar_agent.models.answer import ClaimWithCitations

    final = FinalAnswer(
        markdown=(
            "## Answer\n\n**Question:** What is Self-RAG?\n\n"
            "### Claims\n\n"
            "- Self-RAG retrieves on demand and critiques with reflection tokens "
            "[paper_arxiv_2310_11511 p.1]\n"
        ),
        claims=[
            ClaimWithCitations(
                claim_id="claim_1",
                text="Self-RAG retrieves on demand and critiques with reflection tokens.",
                evidence_ids=["ev_def"],
            )
        ],
        source_cards=[card],
        sources=[card.format_reference()],
        citation_report=CitationReport(
            is_valid=True,
            cited_evidence_ids=["ev_def"],
            cited_paper_ids=["paper_arxiv_2310_11511"],
        ),
    )
    session = DemoSessionResult(
        run_id=run_id,
        query="What is Self-RAG?",
        settings=settings,
        offline_replay=True,
        answer_markdown=final.markdown,
        claims=[c.model_dump(mode="json") for c in final.claims],
        source_cards=[card],
        evidence=[
            EvidenceItem(
                evidence_id="ev_def",
                sub_question_id="sq_0",
                claim="Self-RAG retrieves on demand and critiques with reflection tokens.",
                evidence_text=card.snippet,
                paper_id=card.paper_id,
                chunk_id=card.chunk_id,
                page_start=1,
                page_end=1,
                retrieval_method="hybrid_rerank",
            )
        ],
        plan=QueryPlan(
            original_query="What is Self-RAG?",
            answer_type="factual",
            sub_questions=[
                SubQuestion(
                    id="sq_0",
                    question="What is Self-RAG?",
                    query_type=QueryType.SEMANTIC,
                    status=SubQuestionStatus.COVERED,
                )
            ],
        ),
        verification=VerificationResult(
            is_sufficient=True,
            coverage_score=1.0,
            covered_sub_questions=["sq_0"],
            rationale_summary="Definition evidence found.",
        ),
        final_answer=final,
        events=[
            ExecutionEvent(
                run_id=run_id,
                event_type=EventType.RUN_FINISHED,
                component="workflow",
                summary="workflow finished: evidence_sufficient",
            )
        ],
        trace=TraceSummary(
            query_type="semantic",
            answer_type="factual",
            sub_questions=[
                {
                    "id": "sq_0",
                    "question": "What is Self-RAG?",
                    "status": "covered",
                    "query_type": "semantic",
                }
            ],
            evidence_count=1,
            verified_evidence_count=1,
            coverage_score=1.0,
            is_sufficient=True,
            corrective_iterations=0,
            citation_valid=True,
            terminated_reason="evidence_sufficient",
            latency_ms=180,
            tool_call_count=2,
            token_estimate=400,
            retrieval_methods=["hybrid_rerank"],
        ),
        naive=NaiveComparisonView(
            answer="Self-RAG retrieves on demand… [paper_arxiv_2310_11511 p.1]",
            method="naive_rag:hybrid_rerank",
            hit_count=3,
            latency_ms=40,
        ),
        status={"ok": True, "offline_fixture": True},
    )
    return SavedDemoRun(
        demo_id="what_is_selfrag",
        title="What is Self-RAG? (factual)",
        query=session.query,
        settings=settings,
        created_at=utc_now_iso(),
        offline=True,
        notes="Simple single-paper factual path with page-traceable source card.",
        session=session,
    )


def _fixture_unanswerable() -> SavedDemoRun:
    run_id = "run_demo_unanswerable"
    settings = DemoSettings(compare_naive_rag=True, enable_corrective=True)
    final = FinalAnswer(
        markdown=(
            "## Answer\n\n"
            "**Question:** What was the closing stock price of GraphRAG Inc. yesterday?\n\n"
            "> **Limitation:** The corpus does not contain live financial market prices. "
            "No verified evidence supports an answer.\n"
        ),
        claims=[],
        source_cards=[],
        citation_report=CitationReport(is_valid=True),
        corpus_insufficient=True,
    )
    session = DemoSessionResult(
        run_id=run_id,
        query="What was the closing stock price of GraphRAG Inc. yesterday?",
        settings=settings,
        offline_replay=True,
        answer_markdown=final.markdown,
        final_answer=final,
        verification=VerificationResult(
            is_sufficient=False,
            coverage_score=0.0,
            unanswerable=True,
            rationale_summary="Corpus cannot answer live market questions.",
        ),
        plan=QueryPlan(
            original_query="What was the closing stock price of GraphRAG Inc. yesterday?",
            answer_type="factual",
            sub_questions=[
                SubQuestion(
                    id="sq_0",
                    question="What was the closing stock price of GraphRAG Inc. yesterday?",
                    query_type=QueryType.KEYWORD,
                    status=SubQuestionStatus.MISSING,
                )
            ],
        ),
        events=[
            ExecutionEvent(
                run_id=run_id,
                event_type=EventType.VERIFICATION,
                component="verifier",
                summary="corpus_cannot_answer",
            ),
            ExecutionEvent(
                run_id=run_id,
                event_type=EventType.RUN_FINISHED,
                component="workflow",
                summary="workflow finished: corpus_cannot_answer",
            ),
        ],
        trace=TraceSummary(
            query_type="keyword",
            answer_type="factual",
            evidence_count=0,
            verified_evidence_count=0,
            coverage_score=0.0,
            is_sufficient=False,
            corrective_iterations=1,
            unanswerable=True,
            terminated_reason="corpus_cannot_answer",
            citation_valid=True,
            latency_ms=260,
            tool_call_count=3,
            token_estimate=120,
        ),
        naive=NaiveComparisonView(
            answer="No supporting passages were retrieved… Limitation: corpus may not contain an answer.",
            method="naive_rag:hybrid_rerank",
            hit_count=0,
            latency_ms=30,
        ),
        status={"ok": True, "offline_fixture": True},
    )
    return SavedDemoRun(
        demo_id="unanswerable_market",
        title="Unanswerable market question",
        query=session.query,
        settings=settings,
        created_at=utc_now_iso(),
        offline=True,
        notes="Demonstrates corpus insufficiency / refusal path.",
        session=session,
    )


def try_live(query: str, demo_id: str, title: str, settings: DemoSettings) -> SavedDemoRun | None:
    cfg = load_config()
    setup_logging(cfg)
    status = collect_system_status(cfg)
    if not status.ok:
        return None
    service = DemoService(config=cfg)
    session = service.run_live(query, settings)
    if session.error:
        return None
    return service.to_saved_run(
        session,
        demo_id=demo_id,
        title=title,
        notes="Precomputed from local indexes.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Prefer live index runs when available (still writes fixtures as fallback)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="Output directory for demo JSON runs",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fixtures = [
        (
            "selfrag_vs_crag",
            "Self-RAG vs CRAG (corrective loop)",
            "Compare Self-RAG versus CRAG",
            DemoSettings(compare_naive_rag=True, enable_corrective=True, enable_graph=True),
            _fixture_compare,
        ),
        (
            "what_is_selfrag",
            "What is Self-RAG? (factual)",
            "What is Self-RAG?",
            DemoSettings(compare_naive_rag=True, enable_corrective=False, enable_graph=False),
            _fixture_factual,
        ),
        (
            "unanswerable_market",
            "Unanswerable market question",
            "What was the closing stock price of GraphRAG Inc. yesterday?",
            DemoSettings(compare_naive_rag=True, enable_corrective=True),
            _fixture_unanswerable,
        ),
    ]

    for demo_id, title, query, settings, fixture_fn in fixtures:
        run: SavedDemoRun | None = None
        if args.live:
            run = try_live(query, demo_id, title, settings)
            if run is not None:
                print(f"[live] {demo_id}")
        if run is None:
            run = fixture_fn()
            print(f"[fixture] {demo_id}")
        path = save_demo_run(run, args.out, filename=f"{demo_id}.json")
        print(f"  → {path}")


if __name__ == "__main__":
    main()

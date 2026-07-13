#!/usr/bin/env python3
"""Precompute offline demo runs for interview-safe Streamlit replay.

Preference order:
1. Live run against local indexes (hash embedder) when available.
2. Fall back to curated offline fixtures so the demo works with no indexes.

Writes JSON under data/demo/runs/.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from scholar_agent.app.demo_models import (
    DemoSessionResult,
    DemoSettings,
    NaiveComparisonView,
    SavedDemoRun,
    TraceSummary,
)
from scholar_agent.app.demo_runs import save_demo_run
from scholar_agent.app.demo_service import DemoService, build_corrective_steps
from scholar_agent.app.status import collect_system_status
from scholar_agent.config import load_config
from scholar_agent.ids import make_chunk_id
from scholar_agent.logging import setup_logging
from scholar_agent.models.answer import (
    CitationReport,
    ClaimWithCitations,
    FinalAnswer,
    SourceCard,
)
from scholar_agent.models.base import EventType, ExecutionEvent, QueryType, utc_now_iso
from scholar_agent.models.corpus import Chunk
from scholar_agent.models.evidence import EvidenceItem
from scholar_agent.models.planning import QueryPlan, SubQuestion, SubQuestionStatus
from scholar_agent.models.workflow import VerificationResult
from scholar_agent.retrieval.chunk_store import ChunkStore

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "demo" / "runs"


@dataclass(frozen=True)
class FixtureSource:
    """A passage used by a replay fixture, with an explicit trust level."""

    paper_id: str
    title: str
    pdf_path: str
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    canonical: bool


def _searchable(value: str) -> str:
    """Normalize PDF line wrapping while keeping phrase matching deterministic."""
    dehyphenated = re.sub(r"-\s*\n\s*", "", value)
    return " ".join(re.findall(r"[a-z0-9]+", dehyphenated.casefold()))


def _claim_terms(value: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "in",
        "is",
        "of",
        "on",
        "the",
        "to",
        "uses",
        "with",
    }
    return {token for token in _searchable(value).split() if len(token) > 2 and token not in stop}


def _select_supporting_chunk(
    store: ChunkStore,
    *,
    paper_id: str,
    keywords: tuple[str, ...],
    claim: str,
) -> Chunk:
    """Select a canonical chunk from one paper that supports a fixture claim.

    Every keyword phrase must be present after de-hyphenating PDF line wraps. Among
    qualifying chunks, claim-token coverage wins, with stable chunk ID tie-breaking.
    This intentionally does not accept a remembered chunk ID: rechunking the corpus
    must cause fixtures to follow the new canonical store.
    """
    phrases = tuple(_searchable(keyword) for keyword in keywords if _searchable(keyword))
    if not phrases:
        raise ValueError("at least one non-empty support keyword is required")
    claim_terms = _claim_terms(claim)
    candidates = []
    for chunk in store.chunks:
        if chunk.paper_id != paper_id:
            continue
        searchable = _searchable(chunk.text)
        if not all(phrase in searchable for phrase in phrases):
            continue
        chunk_terms = set(searchable.split())
        coverage = len(claim_terms & chunk_terms) / len(claim_terms) if claim_terms else 0.0
        if claim_terms and coverage < 0.5:
            continue
        candidates.append((coverage, -chunk.token_count, chunk.chunk_id, chunk))
    if not candidates:
        raise ValueError(f"no canonical support chunk for {paper_id} matching {list(keywords)!r}")
    return max(candidates, key=lambda item: item[:3])[3]


def _portable_pdf_path(value: str) -> str:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (REPO / path).resolve()
    try:
        return resolved.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        # Do not bake a developer-specific absolute path into a committed replay.
        return f"data/papers/{path.name}"


def _load_canonical_store() -> ChunkStore | None:
    try:
        return ChunkStore.from_processed_dir(REPO / "data" / "processed")
    except (FileNotFoundError, ValueError):
        return None


def _canonical_source(
    store: ChunkStore,
    *,
    paper_id: str,
    keywords: tuple[str, ...],
    claim: str,
) -> FixtureSource:
    chunk = _select_supporting_chunk(
        store,
        paper_id=paper_id,
        keywords=keywords,
        claim=claim,
    )
    paper = store.get_paper(paper_id)
    if paper is None:
        raise ValueError(f"canonical paper metadata missing: {paper_id}")
    return FixtureSource(
        paper_id=paper_id,
        title=paper.title,
        pdf_path=_portable_pdf_path(paper.pdf_path),
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        canonical=True,
    )


def _unverified_fallback_source(
    *,
    paper_id: str,
    title: str,
    pdf_filename: str,
    text: str,
) -> FixtureSource:
    """Build an explicitly unverified fallback for source-less installations."""
    return FixtureSource(
        paper_id=paper_id,
        title=title,
        pdf_path=f"data/papers/{pdf_filename}",
        chunk_id=make_chunk_id(
            paper_id,
            page_start=1,
            page_end=1,
            text=text,
            section="unverified demo fallback",
        ),
        text=text,
        page_start=1,
        page_end=1,
        canonical=False,
    )


SELF_RAG_CLAIM = "Self-RAG uses reflection tokens to retrieve on demand."
CRAG_CLAIM = "CRAG evaluates retrieved documents and triggers corrective retrieval."


def _fixture_sources() -> tuple[FixtureSource, FixtureSource, str | None, bool]:
    store = _load_canonical_store()
    if store is not None:
        self_rag = _canonical_source(
            store,
            paper_id="paper_arxiv_2310_11511",
            keywords=("SELF-RAG", "reflection tokens", "on-demand"),
            claim=SELF_RAG_CLAIM,
        )
        crag = _canonical_source(
            store,
            paper_id="paper_arxiv_2401_15884",
            keywords=("retrieval evaluator", "confidence degree", "retrieval actions"),
            claim=CRAG_CLAIM,
        )
        return self_rag, crag, store.fingerprint, True

    # These are published abstract excerpts, not canonical corpus claims. They
    # keep a source-less clone replayable, while provenance_verified stays false.
    self_rag_text = (
        "We introduce a new framework called Self-Reflective Retrieval-Augmented "
        "Generation (SELF-RAG) that enhances an LM's quality and factuality through "
        "retrieval and self-reflection. Our framework trains a single arbitrary LM "
        "that adaptively retrieves passages on-demand, and generates and reflects on "
        "retrieved passages and its own generations using special tokens, called "
        "reflection tokens."
    )
    crag_text = (
        "Specifically, a lightweight retrieval evaluator is designed to assess the overall quality "
        "of retrieved documents for a query, returning a confidence degree based on "
        "which different knowledge retrieval actions can be triggered."
    )
    return (
        _unverified_fallback_source(
            paper_id="paper_arxiv_2310_11511",
            title="Self-RAG",
            pdf_filename="2310.11511.pdf",
            text=self_rag_text,
        ),
        _unverified_fallback_source(
            paper_id="paper_arxiv_2401_15884",
            title="CRAG",
            pdf_filename="2401.15884.pdf",
            text=crag_text,
        ),
        None,
        False,
    )


def _citation(source: FixtureSource) -> str:
    page = (
        f"p.{source.page_start}"
        if source.page_start == source.page_end
        else f"pp.{source.page_start}-{source.page_end}"
    )
    return f"[{source.paper_id} {page}]"


def _fixture_compare() -> SavedDemoRun:
    run_id = "run_demo_selfrag_crag"
    self_rag, crag, fingerprint, verified = _fixture_sources()
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
            claim=SELF_RAG_CLAIM,
            evidence_text=self_rag.text,
            paper_id=self_rag.paper_id,
            chunk_id=self_rag.chunk_id,
            page_start=self_rag.page_start,
            page_end=self_rag.page_end,
            retrieval_method="hybrid_rerank",
            retrieval_score=0.91,
        ),
        EvidenceItem(
            evidence_id="ev_crag",
            sub_question_id="sq_1",
            claim=CRAG_CLAIM,
            evidence_text=crag.text,
            paper_id=crag.paper_id,
            chunk_id=crag.chunk_id,
            page_start=crag.page_start,
            page_end=crag.page_end,
            retrieval_method="hybrid_rerank",
            retrieval_score=0.88,
        ),
    ]
    source_by_paper = {source.paper_id: source for source in (self_rag, crag)}
    cards = []
    for item in evidence:
        source = source_by_paper[item.paper_id]
        cards.append(
            SourceCard(
                evidence_id=item.evidence_id,
                paper_id=item.paper_id,
                chunk_id=item.chunk_id,
                page_start=item.page_start,
                page_end=item.page_end,
                snippet=item.evidence_text[:800],
                retrieval_method=item.retrieval_method,
                title=source.title,
                pdf_path=source.pdf_path,
            )
        )
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
            f"- {SELF_RAG_CLAIM} {_citation(self_rag)}\n"
            f"- {CRAG_CLAIM} {_citation(crag)}\n"
        ),
        claims=[
            ClaimWithCitations(
                claim_id="claim_1",
                text=SELF_RAG_CLAIM,
                evidence_ids=["ev_selfrag"],
            ),
            ClaimWithCitations(
                claim_id="claim_2",
                text=CRAG_CLAIM,
                evidence_ids=["ev_crag"],
            ),
        ],
        sources=[c.format_reference() for c in cards],
        source_cards=cards,
        citation_report=CitationReport(
            is_valid=True,
            cited_evidence_ids=["ev_selfrag", "ev_crag"],
            cited_paper_ids=[
                self_rag.paper_id,
                crag.paper_id,
            ],
        ),
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
            event_type=EventType.TOOL_RESULT,
            component="researcher",
            summary="hybrid_rerank found CRAG evidence",
            payload={"method": "hybrid_rerank", "tool_name": "hybrid_search"},
        ),
        ExecutionEvent(
            run_id=run_id,
            event_type=EventType.VERIFICATION,
            component="verifier",
            summary="both comparison sides covered",
            payload={"is_sufficient": True},
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
            corrective_steps=build_corrective_steps(events),
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
        status={
            "ok": True,
            "offline_fixture": True,
            "fixture_provenance": "canonical" if verified else "unverified_fallback",
        },
    )
    return SavedDemoRun(
        demo_id="selfrag_vs_crag",
        title="Self-RAG vs CRAG (corrective loop)",
        query=session.query,
        settings=settings,
        created_at=utc_now_iso(),
        offline=True,
        notes="Shows corrective iteration when one comparison side is initially missing.",
        corpus_fingerprint=fingerprint,
        provenance_verified=verified,
        session=session,
    )


def _fixture_factual() -> SavedDemoRun:
    run_id = "run_demo_what_is_selfrag"
    self_rag, _crag, fingerprint, verified = _fixture_sources()
    settings = DemoSettings(compare_naive_rag=True, enable_graph=False, enable_corrective=False)
    card = SourceCard(
        evidence_id="ev_def",
        paper_id=self_rag.paper_id,
        chunk_id=self_rag.chunk_id,
        page_start=self_rag.page_start,
        page_end=self_rag.page_end,
        snippet=self_rag.text[:800],
        retrieval_method="hybrid_rerank",
        title=self_rag.title,
        pdf_path=self_rag.pdf_path,
    )
    final = FinalAnswer(
        markdown=(
            "## Answer\n\n**Question:** What is Self-RAG?\n\n"
            "### Claims\n\n"
            "- Self-RAG retrieves on demand and critiques with reflection tokens "
            f"{_citation(self_rag)}\n"
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
            cited_paper_ids=[self_rag.paper_id],
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
                evidence_text=self_rag.text,
                paper_id=card.paper_id,
                chunk_id=card.chunk_id,
                page_start=self_rag.page_start,
                page_end=self_rag.page_end,
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
        status={
            "ok": True,
            "offline_fixture": True,
            "fixture_provenance": "canonical" if verified else "unverified_fallback",
        },
    )
    return SavedDemoRun(
        demo_id="what_is_selfrag",
        title="What is Self-RAG? (factual)",
        query=session.query,
        settings=settings,
        created_at=utc_now_iso(),
        offline=True,
        notes="Simple single-paper factual path with page-traceable source card.",
        corpus_fingerprint=fingerprint,
        provenance_verified=verified,
        session=session,
    )


def _fixture_unanswerable() -> SavedDemoRun:
    run_id = "run_demo_unanswerable"
    store = _load_canonical_store()
    fingerprint = store.fingerprint if store is not None else None
    verified = store is not None
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
            corrective_iterations=0,
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
        status={
            "ok": True,
            "offline_fixture": True,
            "fixture_provenance": "canonical" if verified else "unverified_fallback",
        },
    )
    return SavedDemoRun(
        demo_id="unanswerable_market",
        title="Unanswerable market question",
        query=session.query,
        settings=settings,
        created_at=utc_now_iso(),
        offline=True,
        notes="Demonstrates corpus insufficiency / refusal path.",
        corpus_fingerprint=fingerprint,
        provenance_verified=verified,
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

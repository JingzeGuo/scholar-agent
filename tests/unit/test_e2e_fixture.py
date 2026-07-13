"""Offline end-to-end path using five small legal fixture papers.

fixture documents → chunk store → indexes → plan → retrieval → ledger →
verify → write → citation validation → final cited answer

Fixed factual, comparison, relational, and unanswerable questions exercise
dense, hybrid, graph, and corrective retrieval without paid APIs or network calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel, Field

from scholar_agent.agents.citation_validator import CitationValidator
from scholar_agent.agents.planner import Planner
from scholar_agent.agents.researcher import ResearchAgent, ResearchAgentConfig
from scholar_agent.agents.verifier import Verifier
from scholar_agent.agents.workflow import ResearchWorkflow, WorkflowConfig
from scholar_agent.agents.writer import Writer
from scholar_agent.graph.retrieve import GraphRetriever
from scholar_agent.graph.store import KnowledgeGraphStore
from scholar_agent.models.base import QueryType
from scholar_agent.models.corpus import Chunk, Paper
from scholar_agent.models.graph import Entity, Relation
from scholar_agent.models.routing import RetrievalPolicy
from scholar_agent.retrieval.chunk_store import ChunkStore
from scholar_agent.retrieval.dense import DenseIndex
from scholar_agent.retrieval.embeddings import HashingEmbedder
from scholar_agent.retrieval.reranker import LexicalReranker
from scholar_agent.retrieval.router import recommend_policy
from scholar_agent.retrieval.sparse import BM25Index
from scholar_agent.retrieval.tools import RetrievalToolkit
from scholar_agent.storage.jsonl import JsonlRepository


class E2EQuestionFixture(BaseModel):
    """One deterministic question and its externally observable expectations."""

    question_id: str
    kind: Literal["factual", "comparison", "relational", "unanswerable"]
    question: str
    expected_query_type: QueryType
    expected_policy: RetrievalPolicy
    expected_terms: list[str] = Field(default_factory=list)
    expected_paper_ids: list[str] = Field(default_factory=list)
    expect_unanswerable: bool = False


def _load_question_fixtures(repo_root: Path) -> list[E2EQuestionFixture]:
    return JsonlRepository(
        repo_root / "tests" / "fixtures" / "e2e_questions.jsonl",
        E2EQuestionFixture,
    ).read_all()


def _materialize_fixture_pdfs(
    tmp_path: Path,
    papers: list[Paper],
    chunks: list[Chunk],
) -> list[Paper]:
    """Create tiny real PDFs so citation checks include physical page bounds."""
    pdf_dir = tmp_path / "fixture_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    materialized: list[Paper] = []
    for paper in papers:
        paper_chunks = [chunk for chunk in chunks if chunk.paper_id == paper.paper_id]
        page_count = max(chunk.page_end for chunk in paper_chunks)
        pdf_path = pdf_dir / Path(paper.pdf_path).name
        with pymupdf.open() as document:
            for page_number in range(1, page_count + 1):
                page = document.new_page()
                page_text = "\n\n".join(
                    chunk.text
                    for chunk in paper_chunks
                    if chunk.page_start <= page_number <= chunk.page_end
                )
                if page_text:
                    page.insert_textbox(page.rect + (36, 36, -36, -36), page_text)
            document.save(pdf_path)
        materialized.append(
            paper.model_copy(
                update={"pdf_path": str(pdf_path), "page_count": page_count},
            )
        )
    return materialized


def _build_fixture_toolkit(tmp_path: Path, repo_root: Path) -> RetrievalToolkit:
    fixture_dir = repo_root / "tests" / "fixtures"
    papers = JsonlRepository(fixture_dir / "papers.jsonl", Paper).read_all()
    chunks = JsonlRepository(fixture_dir / "chunks.jsonl", Chunk).read_all()
    entities = JsonlRepository(fixture_dir / "graph_entities.jsonl", Entity).read_all()
    relations = JsonlRepository(fixture_dir / "graph_relations.jsonl", Relation).read_all()
    assert papers and chunks, "expected fixture papers and chunks"
    assert len(papers) == 5
    assert {chunk.paper_id for chunk in chunks} == {paper.paper_id for paper in papers}
    papers = _materialize_fixture_pdfs(tmp_path, papers, chunks)

    store = ChunkStore(chunks, papers)
    for relation in relations:
        chunk = store.get_chunk(relation.chunk_id)
        assert chunk is not None
        assert relation.paper_id == chunk.paper_id
        assert relation.page_number == chunk.page_start
        assert relation.page_end == chunk.page_end
        assert relation.evidence_span in chunk.text
    graph_store = KnowledgeGraphStore.from_entities_relations(entities, relations)
    graph = GraphRetriever(graph_store, store, max_hops=2)
    embedder = HashingEmbedder(dimension=64)
    dense = DenseIndex.build(
        store,
        embedder=embedder,
        persist_dir=tmp_path / "chroma",
        collection_name="fixture_e2e",
    )
    sparse = BM25Index.build(store)
    sparse.save(tmp_path / "bm25")
    sparse = BM25Index.load(tmp_path / "bm25", store, verify=True)
    return RetrievalToolkit(
        store,
        dense=dense,
        sparse=sparse,
        graph=graph,
        reranker=LexicalReranker(),
        dense_top_k=4,
        sparse_top_k=4,
        fused_top_k=4,
        rerank_top_k=4,
    )


def _build_fixture_workflow(toolkit: RetrievalToolkit) -> ResearchWorkflow:
    return ResearchWorkflow(
        toolkit,
        config=WorkflowConfig(
            max_corrective_iterations=1,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=3,
                max_iterations_per_pass=3,
                allow_policy_override=True,
            ),
            parallel_research=False,
        ),
        planner=Planner(),
        verifier=Verifier(),
        writer=Writer(),
        citation_validator=CitationValidator(
            provenance_store=toolkit.store,
            require_pdf_provenance=True,
        ),
    )


def test_offline_fixture_corpus_and_indexes_are_traceable(tmp_path: Path, repo_root: Path) -> None:
    toolkit = _build_fixture_toolkit(tmp_path, repo_root)

    # Canonical store: chunks have page provenance
    assert toolkit.store.chunks
    for chunk in toolkit.store.chunks:
        assert chunk.page_start >= 1
        assert chunk.chunk_id

    # Indexes align on stable chunk IDs
    dense_hits = toolkit.dense_search("Self-RAG reflection", k=2).hits
    sparse_hits = toolkit.sparse_search("Self-RAG", k=2).hits
    assert all(h.chunk_id in toolkit.store.by_chunk_id for h in dense_hits + sparse_hits)
    graph_hits = toolkit.graph_search("Which dataset does DPR evaluate on?", k=2).hits
    assert graph_hits
    assert graph_hits[0].chunk_id == "chunk_fixture_dpr_p7"


def test_offline_fixture_e2e_fixed_question_suite(tmp_path: Path, repo_root: Path) -> None:
    """Every fixed question traverses plan → tools → verify → write → citations."""
    toolkit = _build_fixture_toolkit(tmp_path, repo_root)
    workflow = _build_fixture_workflow(toolkit)
    questions = _load_question_fixtures(repo_root)
    assert {case.kind for case in questions} == {
        "factual",
        "comparison",
        "relational",
        "unanswerable",
    }

    for case in questions:
        routing = recommend_policy(case.question, has_graph=True)
        assert routing.query_type == case.expected_query_type
        assert routing.recommended_policy == case.expected_policy

        result = workflow.run(case.question, run_id=f"run_fixture_{case.question_id}")
        assert result.plan is not None
        assert result.plan.sub_questions
        assert result.evidence_ledger is not None
        assert result.verification is not None
        assert result.terminated_reason
        assert result.final_answer is not None
        assert result.final_answer.markdown

        # Page-level provenance on every retrieved item remains canonical.
        for item in result.evidence_ledger.items:
            chunk = toolkit.store.get_chunk(item.chunk_id)
            assert chunk is not None
            assert item.paper_id == chunk.paper_id
            assert item.page_start == chunk.page_start
            assert item.page_end == chunk.page_end

        # The Research Agent actually executed the policy selected by the router.
        policies = {
            str(event.payload.get("recommended_policy"))
            for event in result.events
            if event.component == "router"
        }
        assert case.expected_policy.value in policies
        tool_names = {
            str(event.payload.get("tool_name"))
            for event in result.events
            if event.event_type.value == "tool_selected"
        }
        if case.expected_policy == RetrievalPolicy.DENSE:
            assert "dense_search" in tool_names
        elif case.expected_policy == RetrievalPolicy.GRAPH:
            assert "graph_search" in tool_names
        elif case.expected_policy == RetrievalPolicy.HYBRID_PLUS_GRAPH:
            assert {"hybrid_rerank_search", "graph_search"}.issubset(tool_names)

        if case.expect_unanswerable:
            assert result.unanswerable is True
            assert result.verification.unanswerable is True
            assert result.terminated_reason == "corpus_cannot_answer"
            assert result.final_answer.corpus_insufficient is True
            assert result.final_answer.claims == []
            continue

        assert result.unanswerable is False
        evidence_text = " ".join(
            item.evidence_text for item in result.evidence_ledger.items
        ).casefold()
        for term in case.expected_terms:
            assert term.casefold() in evidence_text
        papers = {item.paper_id for item in result.evidence_ledger.items}
        assert set(case.expected_paper_ids).issubset(papers)
        assert result.final_answer.claims
        assert result.final_answer.citation_report is not None
        assert result.final_answer.citation_report.is_valid


def test_offline_components_compose_without_network(tmp_path: Path, repo_root: Path) -> None:
    toolkit = _build_fixture_toolkit(tmp_path, repo_root)
    plan = Planner().plan("Compare Self-RAG and CRAG")
    assert plan.sub_questions
    agent = ResearchAgent(
        toolkit,
        config=ResearchAgentConfig(max_tool_calls_per_pass=2, max_iterations_per_pass=2),
    )
    research = agent.research_many(
        plan.sub_questions,
        original_query=plan.original_query,
        parallel=False,
    )
    assert research.evidence_ledger is not None
    verification = Verifier().verify(
        query=plan.original_query,
        plan=plan,
        ledger=research.evidence_ledger,
    )
    draft = Writer().write(
        query=plan.original_query,
        plan=plan,
        ledger=research.evidence_ledger,
        verification=verification,
    )
    final = CitationValidator(
        provenance_store=toolkit.store,
        require_pdf_provenance=False,
    ).validate(draft, research.evidence_ledger)
    assert final.markdown
    assert final.citation_report is not None

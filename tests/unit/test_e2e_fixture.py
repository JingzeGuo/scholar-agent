"""Offline end-to-end path using the small legal fixture corpus.

fixture documents → chunk store → indexes → plan → retrieval → ledger →
verify → write → citation validation → final cited answer

No paid APIs or network calls.
"""

from __future__ import annotations

from pathlib import Path

from scholar_agent.agents.citation_validator import CitationValidator
from scholar_agent.agents.planner import Planner
from scholar_agent.agents.researcher import ResearchAgent, ResearchAgentConfig
from scholar_agent.agents.verifier import Verifier
from scholar_agent.agents.workflow import ResearchWorkflow, WorkflowConfig
from scholar_agent.agents.writer import Writer
from scholar_agent.models.corpus import Chunk, Paper
from scholar_agent.retrieval.chunk_store import ChunkStore
from scholar_agent.retrieval.dense import DenseIndex
from scholar_agent.retrieval.embeddings import HashingEmbedder
from scholar_agent.retrieval.reranker import LexicalReranker
from scholar_agent.retrieval.sparse import BM25Index
from scholar_agent.retrieval.tools import RetrievalToolkit
from scholar_agent.storage.jsonl import JsonlRepository


def _build_fixture_toolkit(tmp_path: Path, repo_root: Path) -> RetrievalToolkit:
    fixture_dir = repo_root / "tests" / "fixtures"
    papers = JsonlRepository(fixture_dir / "papers.jsonl", Paper).read_all()
    chunks = JsonlRepository(fixture_dir / "chunks.jsonl", Chunk).read_all()
    assert papers and chunks, "expected fixture papers and chunks"

    store = ChunkStore(chunks, papers)
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
        graph=None,
        reranker=LexicalReranker(),
        dense_top_k=4,
        sparse_top_k=4,
        fused_top_k=4,
        rerank_top_k=4,
    )


def test_offline_fixture_e2e_pipeline(tmp_path: Path, repo_root: Path) -> None:
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

    # Planning → research → verify → write → cite
    wf = ResearchWorkflow(
        toolkit,
        config=WorkflowConfig(
            max_corrective_iterations=1,
            research=ResearchAgentConfig(
                max_tool_calls_per_pass=3,
                max_iterations_per_pass=2,
                allow_policy_override=True,
            ),
            parallel_research=False,
        ),
        planner=Planner(),
        verifier=Verifier(),
        writer=Writer(),
        citation_validator=CitationValidator(
            provenance_store=toolkit.store,
            require_pdf_provenance=False,
        ),
    )
    result = wf.run("What is Self-RAG?")

    assert result.plan is not None
    assert result.plan.sub_questions
    assert result.evidence_ledger is not None
    assert result.verification is not None
    assert result.terminated_reason
    assert result.final_answer is not None
    assert result.final_answer.markdown

    # Page-level provenance on retrieved evidence
    assert result.evidence_ledger.items, "expected at least one evidence item for Self-RAG fixture"
    for item in result.evidence_ledger.items:
        assert item.page_start >= 1
        assert item.chunk_id in toolkit.store.by_chunk_id
        assert item.paper_id
        chunk = toolkit.store.by_chunk_id[item.chunk_id]
        assert item.page_start == chunk.page_start

    # Citations must reference ledger evidence only
    ledger_ids = {e.evidence_id for e in result.evidence_ledger.items}
    for claim in result.final_answer.claims:
        for eid in claim.evidence_ids:
            assert eid in ledger_ids


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

"""Tests for adaptive query router / classifier."""

from __future__ import annotations

from scholar_agent.models.base import QueryType
from scholar_agent.models.routing import RetrievalPolicy
from scholar_agent.retrieval.router import (
    classify_query_type,
    policy_to_tool_modes,
    recommend_policy,
)


def test_semantic_uses_dense() -> None:
    q = "What is retrieval-augmented generation conceptually?"
    qtype, _ = classify_query_type(q)
    assert qtype == QueryType.SEMANTIC
    decision = recommend_policy(q, has_graph=True)
    assert decision.recommended_policy == RetrievalPolicy.DENSE


def test_keyword_acronym_uses_hybrid() -> None:
    q = "Find papers about DPR and BM25 on NQ"
    qtype, signals = classify_query_type(q)
    assert qtype == QueryType.KEYWORD
    assert any("acronym" in s or "known_entities" in s for s in signals)
    decision = recommend_policy(q, has_graph=True)
    assert decision.recommended_policy in {
        RetrievalPolicy.HYBRID,
        RetrievalPolicy.SPARSE,
        RetrievalPolicy.HYBRID_RERANK,
    }


def test_comparison_uses_hybrid_plus_graph() -> None:
    q = "Compare Self-RAG versus CRAG"
    qtype, _ = classify_query_type(q)
    assert qtype == QueryType.COMPARISON
    decision = recommend_policy(q, has_graph=True)
    assert decision.recommended_policy == RetrievalPolicy.HYBRID_PLUS_GRAPH
    modes = policy_to_tool_modes(decision.recommended_policy)
    assert "hybrid_rerank" in modes and "graph" in modes


def test_relational_uses_graph_when_available() -> None:
    q = "Which datasets does Self-RAG evaluate on?"
    qtype, _ = classify_query_type(q)
    assert qtype == QueryType.RELATIONAL
    with_graph = recommend_policy(q, has_graph=True)
    assert with_graph.recommended_policy == RetrievalPolicy.GRAPH
    without = recommend_policy(q, has_graph=False)
    assert without.recommended_policy != RetrievalPolicy.GRAPH


def test_synthesis_uses_hybrid_rerank() -> None:
    q = "Summarize the main trends across agentic RAG papers"
    qtype, _ = classify_query_type(q)
    assert qtype == QueryType.SYNTHESIS
    decision = recommend_policy(q)
    assert decision.recommended_policy == RetrievalPolicy.HYBRID_RERANK


def test_general_evidence_question_uses_hybrid_rerank() -> None:
    query = "Provide evidence supporting retrieval quality improvements"
    query_type, signals = classify_query_type(query)
    assert query_type == QueryType.SEMANTIC
    assert "default_general_evidence" in signals
    assert recommend_policy(query).recommended_policy == RetrievalPolicy.HYBRID_RERANK


def test_corrective_routes_from_missing_aspect() -> None:
    decision = recommend_policy(
        "need more evidence",
        corrective=True,
        missing_aspect="Which metric does DPR report on Natural Questions?",
        has_graph=True,
    )
    assert "corrective" in decision.signals
    assert isinstance(decision.recommended_policy, RetrievalPolicy)
    # Relational/keyword aspect should not be pure semantic dense-only
    assert decision.query_type in {
        QueryType.RELATIONAL,
        QueryType.KEYWORD,
        QueryType.SEMANTIC,
        QueryType.COMPARISON,
        QueryType.SYNTHESIS,
    }


def test_different_query_types_choose_different_tools() -> None:
    policies = {
        recommend_policy("What is RAPTOR?", has_graph=True).recommended_policy,
        recommend_policy("Compare Self-RAG and CRAG", has_graph=True).recommended_policy,
        recommend_policy("Which dataset does DPR evaluate on?", has_graph=True).recommended_policy,
    }
    assert len(policies) >= 2

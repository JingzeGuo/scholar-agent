"""Retrieval stack: dense, sparse BM25, RRF hybrid, reranker, Naive RAG."""

from scholar_agent.retrieval.fusion import reciprocal_rank_fusion
from scholar_agent.retrieval.naive_rag import NaiveRAG
from scholar_agent.retrieval.tools import RetrievalToolkit

__all__ = [
    "NaiveRAG",
    "RetrievalToolkit",
    "reciprocal_rank_fusion",
]

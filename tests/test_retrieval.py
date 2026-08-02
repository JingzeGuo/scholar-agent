from __future__ import annotations

import numpy as np
import pytest

import scholar_agent.indexes as indexes_module
import scholar_agent.reranker as reranker_module
from scholar_agent.graph_store import build_graph, extract_entities, graph_search
from scholar_agent.indexes import (
    BM25Index,
    DenseIndex,
    ModelUnavailableError,
    resolve_model_path,
)
from scholar_agent.reranker import rerank
from scholar_agent.retrieval import reciprocal_rank_fusion


def test_bm25_returns_relevant_result(sample_chunks: list[dict]) -> None:
    results = BM25Index(sample_chunks).search(["reflection tokens"], top_k=3)

    assert results[0]["chunk_id"] == "self-1"
    assert results[0]["score"] > 0


def test_dense_retrieval_uses_cosine_similarity(sample_chunks: list[dict]) -> None:
    embeddings = np.eye(3, dtype=np.float32)
    dense = DenseIndex(sample_chunks, embeddings, "test", "sentence-transformers")
    dense._encode_queries = lambda queries: np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)  # type: ignore[method-assign]

    results = dense.search(["corrective retrieval"], top_k=2)

    assert results[0]["chunk_id"] == "crag-1"
    assert results[0]["score"] == 1.0


def test_dense_index_does_not_fall_back_when_model_is_unavailable(
    sample_chunks: list[dict],
    monkeypatch,
) -> None:
    def fail(texts: list[str], model_name: str) -> np.ndarray:
        raise ModelUnavailableError("download failed")

    monkeypatch.setattr(indexes_module, "_sentence_embeddings", fail)

    with pytest.raises(ModelUnavailableError, match="download failed"):
        DenseIndex.build(sample_chunks, "missing-model")

    with pytest.raises(ModelUnavailableError, match="unsupported fallback backend"):
        DenseIndex(sample_chunks, np.eye(3), "test", "hash")


def test_model_resolution_downloads_or_raises(monkeypatch) -> None:
    downloaded: list[str] = []

    def download(model_name: str) -> str:
        downloaded.append(model_name)
        return "/model-cache"

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert resolve_model_path("org/model") == "/model-cache"
    assert downloaded == ["org/model"]

    def fail(model_name: str) -> str:
        raise OSError("network unavailable")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)
    with pytest.raises(ModelUnavailableError, match="download failed"):
        resolve_model_path("org/missing-model")


def test_graph_retrieval_finds_entity_chunk(sample_chunks: list[dict]) -> None:
    graph = build_graph(sample_chunks)

    results = graph_search(["Self-RAG"], graph, sample_chunks)

    assert extract_entities("Self-RAG and SELF-RAG") == ["self-rag"]
    assert graph.nodes["self-rag"] == {"chunks": ["self-1"]}
    assert results
    assert results[0]["chunk_id"] == "self-1"
    assert results[0]["paper"] == "Self-RAG.pdf"


def test_graph_short_substring_does_not_match_crag(sample_chunks: list[dict]) -> None:
    graph = build_graph(sample_chunks)

    assert graph_search(["RAG"], graph, sample_chunks) == []


def test_graph_normalizes_multiword_entity_alias(sample_chunks: list[dict]) -> None:
    graph = build_graph(sample_chunks)

    results = graph_search(["Self RAG"], graph, sample_chunks)

    assert results
    assert results[0]["chunk_id"] == "self-1"


def test_rrf_rewards_results_found_by_multiple_retrievers(sample_chunks: list[dict]) -> None:
    sparse = [sample_chunks[0], sample_chunks[1]]
    dense = [sample_chunks[1], sample_chunks[2]]
    graph = [sample_chunks[1]]

    fused = reciprocal_rank_fusion(sparse, dense, graph)

    assert [item["chunk_id"] for item in fused] == ["crag-1", "self-1", "other-1"]


def test_reranker_reorders_candidates(sample_chunks: list[dict]) -> None:
    def fake_scorer(pairs: list[tuple[str, str]]) -> list[float]:
        assert len(pairs) == 6
        return [0.1, 0.2, 0.9, 0.1, 0.2, 0.3]

    ranked = rerank(["query one", "query two"], sample_chunks, "unused", scorer=fake_scorer)

    assert ranked[0]["chunk_id"] == "crag-1"
    assert ranked[0]["score"] == 0.9


def test_reranker_does_not_fall_back_when_model_is_unavailable(
    sample_chunks: list[dict],
    monkeypatch,
) -> None:
    def fail(model_name: str) -> object:
        raise ModelUnavailableError("download failed")

    monkeypatch.setattr(reranker_module, "_cross_encoder", fail)

    with pytest.raises(ModelUnavailableError, match="download failed"):
        rerank(["query"], sample_chunks, "missing-model")

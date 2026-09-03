from __future__ import annotations

import numpy as np
import pytest

import scholar_agent.indexes as indexes_module
import scholar_agent.reranker as reranker_module
from scholar_agent.indexes import (
    BM25Index,
    DenseIndex,
    ModelUnavailableError,
    _embedding_model,
    _sentence_embeddings,
    resolve_model_path,
)
from scholar_agent.reranker import rerank
from scholar_agent.retrieval import RetrievalEngine, reciprocal_rank_fusion


def test_bm25_returns_relevant_result(sample_chunks: list[dict]) -> None:
    results = BM25Index(sample_chunks).search(["reflection tokens"], top_k=3)

    assert results[0]["chunk_id"] == "self-1"
    assert results[0]["score"] > 0


def test_dense_search_returns_semantic_matches_and_encodes_query_batch_once(
    sample_chunks: list[dict],
) -> None:
    embeddings = np.eye(3, dtype=np.float32)
    dense = DenseIndex(sample_chunks, embeddings, "test", "sentence-transformers")
    encoded: list[list[str]] = []

    def encode(queries: list[str]) -> np.ndarray:
        encoded.append(queries)
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )

    dense._encode_queries = encode  # type: ignore[method-assign]

    rankings = dense.search_many(["reflection", "corrective"], top_k=2)

    assert encoded == [["reflection", "corrective"]]
    assert [ranking[0]["chunk_id"] for ranking in rankings] == ["self-1", "crag-1"]
    assert rankings[1][0]["score"] == 1.0


def test_indexes_reject_a_different_same_size_corpus(
    sample_chunks: list[dict],
    tmp_path,
) -> None:
    bm25 = BM25Index(sample_chunks)
    bm25.save(tmp_path / "bm25.json")
    dense = DenseIndex(
        sample_chunks,
        np.eye(len(sample_chunks), dtype=np.float32),
        "test",
        "sentence-transformers",
    )
    dense.save(tmp_path)

    changed_chunks = [
        {**chunk, "chunk_id": f"replacement-{index}"}
        for index, chunk in enumerate(sample_chunks)
    ]

    with pytest.raises(ValueError, match="BM25 index does not match the corpus"):
        BM25Index.load(changed_chunks, tmp_path / "bm25.json")
    with pytest.raises(ValueError, match="Dense index does not match the corpus"):
        DenseIndex.load(changed_chunks, tmp_path)


def test_retrieval_engine_uses_fixed_per_query_candidate_limit(
    sample_chunks: list[dict],
    monkeypatch,
) -> None:
    bm25 = BM25Index(sample_chunks)
    dense = DenseIndex(
        sample_chunks,
        np.eye(len(sample_chunks), dtype=np.float32),
        "test",
        "sentence-transformers",
    )
    limits: list[int] = []

    def sparse_search(queries: list[str], top_k: int) -> list[dict]:
        limits.append(top_k)
        return []

    def dense_search(queries: list[str], top_k: int) -> list[list[dict]]:
        limits.append(top_k)
        return [[] for _ in queries]

    monkeypatch.setattr(bm25, "search", sparse_search)
    monkeypatch.setattr(dense, "search_many", dense_search)
    engine = RetrievalEngine(sample_chunks, bm25, dense)

    engine.sparse_search(["reflection"])
    engine.dense_search_many(["reflection"])

    assert limits == [8, 8]


def test_embedding_model_is_cached_by_name(monkeypatch) -> None:
    constructed: list[tuple[str, bool]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_path: str, *, local_files_only: bool) -> None:
            constructed.append((model_path, local_files_only))

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(indexes_module, "resolve_model_path", lambda name: f"/models/{name}")
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        FakeSentenceTransformer,
    )
    _embedding_model.cache_clear()
    try:
        _sentence_embeddings(["first"], "org/model")
        _sentence_embeddings(["second"], "org/model")
    finally:
        _embedding_model.cache_clear()

    assert constructed == [("/models/org/model", True)]


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


def test_rrf_rewards_chunks_found_by_both_rankings(sample_chunks: list[dict]) -> None:
    sparse = [sample_chunks[0], sample_chunks[1]]
    dense = [sample_chunks[1], sample_chunks[2]]

    fused = reciprocal_rank_fusion(sparse, dense)

    assert [item["chunk_id"] for item in fused] == ["crag-1", "self-1", "other-1"]


def test_reranker_reorders_candidates(sample_chunks: list[dict]) -> None:
    def fake_scorer(pairs: list[tuple[str, str]]) -> list[float]:
        assert len(pairs) == 6
        return [0.1, 0.2, 0.9, 0.1, 0.2, 0.3]

    ranked = rerank(["query one", "query two"], sample_chunks, "unused", scorer=fake_scorer)

    assert ranked[0]["chunk_id"] == "crag-1"
    assert ranked[0]["score"] == 0.9
    assert ranked[0]["_query_scores"] == [0.9, 0.1]


def test_reranker_does_not_fall_back_when_model_is_unavailable(
    sample_chunks: list[dict],
    monkeypatch,
) -> None:
    def fail(model_name: str) -> object:
        raise ModelUnavailableError("download failed")

    monkeypatch.setattr(reranker_module, "_cross_encoder", fail)

    with pytest.raises(ModelUnavailableError, match="download failed"):
        rerank(["query"], sample_chunks, "missing-model")

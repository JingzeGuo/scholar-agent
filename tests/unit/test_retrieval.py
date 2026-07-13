"""Phase 3 retrieval unit tests (offline; hashing embedder + lexical reranker)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sentence_transformers

from scholar_agent.config import AppConfig, DenseRetrievalConfig, PathsConfig, RetrievalConfig
from scholar_agent.graph.extract import EXTRACTION_CACHE_SCHEMA
from scholar_agent.graph.pipeline import GRAPH_BUILD_SCHEMA
from scholar_agent.ids import content_hash, make_chunk_id
from scholar_agent.models.corpus import Chunk, Paper
from scholar_agent.models.retrieval import RetrievalFilters
from scholar_agent.retrieval.chunk_store import ChunkStore
from scholar_agent.retrieval.dense import DenseIndex
from scholar_agent.retrieval.embeddings import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    model_cache_folder,
)
from scholar_agent.retrieval.fusion import ranks_map, reciprocal_rank_fusion
from scholar_agent.retrieval.index_builder import build_indexes, load_toolkit
from scholar_agent.retrieval.naive_rag import NaiveRAG
from scholar_agent.retrieval.reranker import CrossEncoderReranker, LexicalReranker
from scholar_agent.retrieval.sparse import BM25Index, tokenize
from scholar_agent.retrieval.tools import RetrievalToolkit
from scholar_agent.storage.jsonl import JsonlRepository


def _chunk(paper_id: str, text: str, page: int, section: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(
            paper_id, page_start=page, page_end=page, text=text, section=section
        ),
        paper_id=paper_id,
        text=text,
        page_start=page,
        page_end=page,
        section=section,
        token_count=len(text.split()),
        content_hash=content_hash(text),
    )


def _fixture_store() -> ChunkStore:
    chunks = [
        _chunk(
            "paper_self_rag",
            "Self-RAG retrieves passages on demand and critiques generation with reflection tokens.",
            3,
            "Method",
        ),
        _chunk(
            "paper_crag",
            "Corrective RAG evaluates retrieved documents and triggers corrective retrieval when quality is low.",
            2,
            "Approach",
        ),
        _chunk(
            "paper_dpr",
            "Dense Passage Retrieval uses dual-encoders for open-domain question answering over Wikipedia.",
            1,
            "Introduction",
        ),
        _chunk(
            "paper_react",
            "ReAct interleaves reasoning traces and task-specific actions for tool-using language agents.",
            4,
            "Method",
        ),
        _chunk(
            "paper_bm25",
            "BM25 is a classic sparse ranking function using term frequency and inverse document frequency.",
            1,
            "Background",
        ),
    ]
    papers = [
        Paper(
            paper_id=pid,
            title=pid,
            pdf_path=f"{pid}.pdf",
            content_hash=content_hash(pid),
        )
        for pid in {c.paper_id for c in chunks}
    ]
    return ChunkStore(chunks, papers)


class _FakeSentenceEmbedder:
    def __init__(self, model_name: str, dimension: int = 12) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        vector[len(text) % self.dimension] = 1.0
        return vector


def _index_config(tmp_path: Path, store: ChunkStore, *, model_name: str) -> AppConfig:
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    JsonlRepository(processed / "chunks.jsonl", Chunk).write_all(store.chunks)
    JsonlRepository(processed / "papers.jsonl", Paper).write_all(store.papers)
    return AppConfig(
        paths=PathsConfig(
            data_dir=tmp_path,
            papers_dir=tmp_path / "papers",
            processed_dir=processed,
            indexes_dir=tmp_path / "indexes",
            evaluation_dir=tmp_path / "evaluation",
            corpus_manifest=tmp_path / "manifest.jsonl",
        ),
        retrieval=RetrievalConfig(
            dense=DenseRetrievalConfig(model_name=model_name, collection_name="test_config")
        ),
        repo_root=tmp_path,
    )


def test_rrf_prefers_shared_top_docs() -> None:
    dense = ["a", "b", "c"]
    sparse = ["b", "d", "a"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] == "b"  # rank1 sparse + rank2 dense
    assert set(ids) == {"a", "b", "c", "d"}
    # scores positive and sorted
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_dedupes_within_list() -> None:
    fused = reciprocal_rank_fusion([["a", "a", "b"]], k=10)
    assert [i for i, _ in fused] == ["a", "b"]


def test_ranks_map() -> None:
    assert ranks_map(["x", "y", "x"]) == {"x": 1, "y": 2}


def test_bm25_index_roundtrip_and_chunk_ids(tmp_path: Path) -> None:
    store = _fixture_store()
    index = BM25Index.build(store)
    index.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25", store, verify=True)
    assert loaded.meta.chunk_ids == store.ordered_ids()
    assert loaded.meta.corpus_fingerprint == store.fingerprint

    hits = loaded.search("corrective retrieval documents", k=3)
    assert hits
    assert all(h.chunk_id in store.by_chunk_id for h in hits)
    assert hits[0].paper_id == "paper_crag"
    assert hits[0].page_start == 2


def test_bm25_fingerprint_mismatch(tmp_path: Path) -> None:
    store = _fixture_store()
    index = BM25Index.build(store)
    index.save(tmp_path / "bm25")
    other = ChunkStore(store.chunks[1:], store.papers)
    try:
        BM25Index.load(tmp_path / "bm25", other, verify=True)
        raise AssertionError("expected fingerprint mismatch")
    except ValueError as exc:
        assert "fingerprint" in str(exc).lower()


def test_dense_index_stable_ids(tmp_path: Path) -> None:
    store = _fixture_store()
    embedder = HashingEmbedder(dimension=32)
    DenseIndex.build(
        store,
        embedder=embedder,
        persist_dir=tmp_path / "chroma",
        collection_name="test_chunks",
    )
    loaded = DenseIndex.load(
        tmp_path / "chroma",
        store,
        embedder=embedder,
        verify=True,
        collection_name="test_chunks",
    )
    hits = loaded.search("Self-RAG reflection tokens retrieve", k=2)
    assert hits
    assert hits[0].chunk_id in store.by_chunk_id
    assert hits[0].retrieval_method == "dense"


def test_hash_dense_search_is_exact_and_stably_tie_broken(tmp_path: Path) -> None:
    chunks = [
        _chunk("paper_b", "identical retrieval passage", 1),
        _chunk("paper_a", "identical retrieval passage", 1),
    ]
    papers = [
        Paper(
            paper_id=paper_id,
            title=paper_id,
            pdf_path=f"{paper_id}.pdf",
            content_hash=content_hash(paper_id),
        )
        for paper_id in ("paper_a", "paper_b")
    ]
    store = ChunkStore(chunks, papers)
    index = DenseIndex.build(
        store,
        embedder=HashingEmbedder(dimension=32),
        persist_dir=tmp_path / "exact_hash",
        collection_name="exact_hash",
    )

    # The deterministic hash path must not delegate tied results to HNSW.
    def unexpected_ann_query(**_kwargs: object) -> object:
        raise AssertionError("hash search should use exact persisted vectors")

    index._collection.query = unexpected_ann_query
    first = index.search("identical retrieval passage", k=2)
    second = index.search("identical retrieval passage", k=2)

    expected = sorted(chunk.chunk_id for chunk in chunks)
    assert [hit.chunk_id for hit in first] == expected
    assert [hit.chunk_id for hit in second] == expected
    assert first[0].score == pytest.approx(first[1].score)


def test_index_builder_rebuilds_when_requested_embedder_changes(tmp_path: Path) -> None:
    store = _fixture_store()
    cfg = _index_config(tmp_path, store, model_name="fake-a")
    shared = tmp_path / "shared-index"
    first = build_indexes(
        config=cfg,
        indexes_dir=shared,
        embedder=_FakeSentenceEmbedder("fake-a", dimension=12),
    )
    assert first.dense.meta["embedder"] == "fake-a"

    second = build_indexes(
        config=cfg,
        indexes_dir=shared,
        embedder=_FakeSentenceEmbedder("fake-b", dimension=16),
    )
    assert second.dense.meta["embedder"] == "fake-b"
    assert second.dense.meta["dimension"] == 16


def test_index_builder_honors_changed_config_model_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _fixture_store()
    cfg_a = _index_config(tmp_path, store, model_name="config-model-a")

    def fake_factory(model_name: str, *, backend: str):
        assert backend == "st"
        return _FakeSentenceEmbedder(model_name)

    monkeypatch.setattr("scholar_agent.retrieval.index_builder.create_embedder", fake_factory)
    first = build_indexes(config=cfg_a, embedding_backend="st")
    assert first.dense.meta["embedder"] == "config-model-a"

    cfg_b = cfg_a.model_copy(
        update={
            "retrieval": cfg_a.retrieval.model_copy(
                update={
                    "dense": cfg_a.retrieval.dense.model_copy(
                        update={"model_name": "config-model-b"}
                    )
                }
            )
        }
    )
    second = build_indexes(config=cfg_b, embedding_backend="st")
    assert second.dense.meta["embedder"] == "config-model-b"


def test_hash_backend_uses_isolated_default_index_directory(tmp_path: Path) -> None:
    store = _fixture_store()
    cfg = _index_config(tmp_path, store, model_name="production-model")
    built = build_indexes(config=cfg, embedding_backend="hash")
    assert built.dense_dir == tmp_path / "indexes" / "hash" / "chroma"
    assert built.sparse_dir == tmp_path / "indexes" / "hash" / "bm25"
    assert not (tmp_path / "indexes" / "chroma" / "meta.json").exists()


def test_load_toolkit_disables_graph_with_stale_corpus_identity(tmp_path: Path) -> None:
    store = _fixture_store()
    cfg = _index_config(tmp_path, store, model_name="production-model")
    processed = Path(cfg.paths.processed_dir)
    (processed / "knowledge_graph.json").write_text("{}\n", encoding="utf-8")
    (processed / "graph_meta.json").write_text(
        f'{{"graph_schema":"{GRAPH_BUILD_SCHEMA}",'
        '"corpus_fingerprint":"stale",'
        f'"extraction_schema":"{EXTRACTION_CACHE_SCHEMA}",'
        '"limit_chunks":null,"use_llm_resolution":false,'
        '"max_llm_resolutions":0}\n',
        encoding="utf-8",
    )

    toolkit = load_toolkit(config=cfg, embedding_backend="hash", load_graph=True)

    assert toolkit.graph is None


def test_hybrid_rrf_and_rerank_debug() -> None:
    store = _fixture_store()
    embedder = HashingEmbedder(dimension=32)
    dense = DenseIndex.build(store, embedder=embedder, collection_name="hybrid_test")
    sparse = BM25Index.build(store)
    toolkit = RetrievalToolkit(
        store,
        dense=dense,
        sparse=sparse,
        reranker=LexicalReranker(),
        dense_top_k=3,
        sparse_top_k=3,
        fused_top_k=4,
        rerank_top_k=3,
    )
    result = toolkit.hybrid_search("BM25 sparse ranking term frequency", rerank=True)
    assert result.method == "hybrid_rerank"
    assert result.hits
    assert all(h.chunk_id in store.by_chunk_id for h in result.hits)
    # debug includes component ranks
    assert "dense_ids" in result.debug
    assert "sparse_ids" in result.debug
    assert result.traces
    assert result.traces[0].final_rank == 1


def test_filters_restrict_paper() -> None:
    store = _fixture_store()
    sparse = BM25Index.build(store)
    hits = sparse.search(
        "retrieval",
        k=10,
        filters=RetrievalFilters(paper_ids=["paper_dpr"]),
    )
    assert hits
    assert all(h.paper_id == "paper_dpr" for h in hits)


def test_naive_rag_includes_page_references() -> None:
    store = _fixture_store()
    embedder = HashingEmbedder(dimension=32)
    dense = DenseIndex.build(store, embedder=embedder, collection_name="naive_test")
    sparse = BM25Index.build(store)
    toolkit = RetrievalToolkit(
        store,
        dense=dense,
        sparse=sparse,
        reranker=LexicalReranker(),
    )
    rag = NaiveRAG(toolkit, llm=None, mode="hybrid_rerank", top_k=3)
    answer = rag.answer("What is Self-RAG?")
    assert answer.citations
    assert answer.hits
    # page references present
    assert any("p." in c.format_inline() for c in answer.citations)
    assert any(
        c.format_inline() in answer.answer or "Sources:" in answer.answer for c in answer.citations
    )
    assert all(c.page_start >= 1 for c in answer.citations)


def test_get_chunk_and_paper_tools(tmp_path: Path) -> None:
    store = _fixture_store()
    # persist canonical store as would exist post-ingest
    proc = tmp_path / "processed"
    proc.mkdir()
    JsonlRepository(proc / "chunks.jsonl", Chunk).write_all(store.chunks)
    JsonlRepository(proc / "papers.jsonl", Paper).write_all(store.papers)
    loaded = ChunkStore.from_processed_dir(proc)
    toolkit = RetrievalToolkit(loaded)
    cid = store.chunks[0].chunk_id
    assert toolkit.get_chunk(cid) is not None
    assert toolkit.get_paper(store.chunks[0].paper_id) is not None


def test_tokenize_basic() -> None:
    assert "retrieval" in tokenize("Retrieval-Augmented Generation")


def test_model_cache_defaults_to_ignored_project_cache(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
) -> None:
    monkeypatch.delenv("SCHOLAR_MODEL_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert Path(model_cache_folder()) == repo_root / ".cache" / "huggingface" / "hub"


def test_model_cache_honors_hf_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SCHOLAR_MODEL_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert Path(model_cache_folder()) == tmp_path / "hub"


def test_production_models_try_local_cache_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeSentenceTransformer:
        def __init__(self, _name: str, **kwargs: object) -> None:
            calls.append(("embedding", bool(kwargs.get("local_files_only"))))

        def encode(self, _texts: list[str], **_kwargs: object) -> np.ndarray:
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    class FakeCrossEncoder:
        def __init__(self, _name: str, **kwargs: object) -> None:
            calls.append(("reranker", bool(kwargs.get("local_files_only"))))

    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)

    SentenceTransformerEmbedder("fixture-embedder")
    CrossEncoderReranker("fixture-reranker")
    assert calls == [("embedding", True), ("reranker", True)]

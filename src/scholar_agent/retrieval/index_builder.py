"""Build and load dense + BM25 indexes from the canonical chunk store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scholar_agent.config import AppConfig, load_config
from scholar_agent.logging import get_logger
from scholar_agent.retrieval.chunk_store import ChunkStore
from scholar_agent.retrieval.dense import DenseIndex
from scholar_agent.retrieval.embeddings import Embedder, create_embedder
from scholar_agent.retrieval.reranker import Reranker, create_reranker
from scholar_agent.retrieval.sparse import BM25Index
from scholar_agent.retrieval.tools import RetrievalToolkit

logger = get_logger(__name__)


@dataclass
class BuiltIndexes:
    store: ChunkStore
    dense: DenseIndex
    sparse: BM25Index
    toolkit: RetrievalToolkit
    dense_dir: Path
    sparse_dir: Path


def build_indexes(
    *,
    config: AppConfig | None = None,
    processed_dir: Path | str | None = None,
    indexes_dir: Path | str | None = None,
    embedder: Embedder | None = None,
    embedding_backend: Literal["auto", "hash", "st"] = "auto",
    force: bool = False,
) -> BuiltIndexes:
    cfg = config or load_config()
    processed = Path(processed_dir or cfg.paths.processed_dir)
    indexes = Path(indexes_dir or cfg.paths.indexes_dir)
    dense_dir = indexes / "chroma"
    sparse_dir = indexes / "bm25"

    store = ChunkStore.from_processed_dir(processed)
    logger.info(
        "building indexes for %s chunks (fingerprint=%s…)",
        len(store),
        store.fingerprint[:12],
    )

    if embedder is None:
        model_name = cfg.retrieval.dense.model_name
        if embedding_backend == "hash":
            model_name = "hashing-embedder-v1"
        embedder = create_embedder(model_name, backend=embedding_backend)

    # BM25
    if (
        not force
        and (sparse_dir / "meta.json").is_file()
        and (sparse_dir / "bm25.pkl").is_file()
    ):
        try:
            sparse = BM25Index.load(sparse_dir, store, verify=True)
            logger.info("loaded existing BM25 index")
        except ValueError:
            logger.info("BM25 fingerprint mismatch; rebuilding")
            sparse = BM25Index.build(store)
            sparse.save(sparse_dir)
    else:
        sparse = BM25Index.build(store)
        sparse.save(sparse_dir)
        logger.info("wrote BM25 index → %s", sparse_dir)

    # Dense
    dense_meta = dense_dir / "meta.json"
    if not force and dense_meta.is_file():
        try:
            dense = DenseIndex.load(
                dense_dir,
                store,
                embedder=embedder,
                verify=True,
                collection_name=cfg.retrieval.dense.collection_name,
            )
            logger.info("loaded existing dense index")
        except (ValueError, FileNotFoundError, Exception) as exc:
            logger.info("dense reload failed (%s); rebuilding", exc)
            dense = DenseIndex.build(
                store,
                embedder=embedder,
                persist_dir=dense_dir,
                collection_name=cfg.retrieval.dense.collection_name,
            )
    else:
        dense = DenseIndex.build(
            store,
            embedder=embedder,
            persist_dir=dense_dir,
            collection_name=cfg.retrieval.dense.collection_name,
        )
        logger.info("wrote dense index → %s", dense_dir)

    toolkit = RetrievalToolkit(
        store,
        dense=dense,
        sparse=sparse,
        dense_top_k=cfg.retrieval.dense.top_k,
        sparse_top_k=cfg.retrieval.sparse.top_k,
        fused_top_k=cfg.retrieval.fusion.fused_top_k,
        rerank_top_k=cfg.retrieval.reranker.top_k,
        rrf_k=cfg.retrieval.fusion.k,
    )
    return BuiltIndexes(
        store=store,
        dense=dense,
        sparse=sparse,
        toolkit=toolkit,
        dense_dir=dense_dir,
        sparse_dir=sparse_dir,
    )


def load_toolkit(
    *,
    config: AppConfig | None = None,
    embedding_backend: Literal["auto", "hash", "st"] = "auto",
    reranker_backend: Literal["auto", "lexical", "cross-encoder"] = "auto",
) -> RetrievalToolkit:
    """Load indexes from disk (rebuild sparse/dense if missing)."""
    cfg = config or load_config()
    built = build_indexes(
        config=cfg,
        embedding_backend=embedding_backend,
        force=False,
    )
    reranker: Reranker
    if reranker_backend == "lexical":
        reranker = create_reranker("lexical-overlap-v1")
    elif reranker_backend == "cross-encoder":
        reranker = create_reranker(cfg.retrieval.reranker.model_name)
    else:
        # auto: prefer configured model, fall back to lexical on import failure
        try:
            if embedding_backend == "hash":
                reranker = create_reranker("lexical-overlap-v1")
            else:
                reranker = create_reranker(cfg.retrieval.reranker.model_name)
        except Exception:  # noqa: BLE001
            logger.warning("cross-encoder unavailable; using lexical reranker")
            reranker = create_reranker("lexical-overlap-v1")
    built.toolkit.reranker = reranker
    return built.toolkit

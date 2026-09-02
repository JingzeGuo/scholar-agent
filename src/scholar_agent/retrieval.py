"""BM25, dense retrieval, and reciprocal rank fusion."""

from __future__ import annotations

from scholar_agent.config import Settings
from scholar_agent.indexes import BM25Index, DenseIndex
from scholar_agent.models import load_chunks

PER_QUERY_CANDIDATES = 8


class RetrievalEngine:
    """Load and call the two indexes used by the fixed hybrid retrieval path."""

    def __init__(
        self,
        chunks: list[dict],
        bm25: BM25Index,
        dense: DenseIndex,
    ) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense

    @classmethod
    def load(cls, settings: Settings) -> RetrievalEngine:
        chunks = load_chunks(settings.chunks_path)
        bm25 = BM25Index.load(chunks, settings.index_dir / "bm25.json")
        dense = DenseIndex.load(chunks, settings.index_dir)
        return cls(chunks, bm25, dense)

    def sparse_search(self, queries: list[str]) -> list[dict]:
        return self.bm25.search(queries, PER_QUERY_CANDIDATES)

    def dense_search_many(self, queries: list[str]) -> list[list[dict]]:
        return self.dense.search_many(queries, PER_QUERY_CANDIDATES)


def reciprocal_rank_fusion(*result_lists: list[dict], k: int = 60) -> list[dict]:
    """Fuse ranks by chunk ID; no weights, contribution models, or factories."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            items[chunk_id] = item
    return sorted(
        items.values(),
        key=lambda item: scores[item["chunk_id"]],
        reverse=True,
    )


def build_all_indexes(settings: Settings) -> dict[str, object]:
    """Build BM25 and dense indexes and return display-only summary values."""
    chunks = load_chunks(settings.chunks_path)
    bm25 = BM25Index(chunks)
    bm25.save(settings.index_dir / "bm25.json")

    dense = DenseIndex.build(chunks, settings.embedding_model)
    dense.save(settings.index_dir)

    return {
        "chunks": len(chunks),
        "dense_backend": dense.backend,
    }

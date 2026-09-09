"""BM25, dense retrieval, and reciprocal rank fusion."""

from __future__ import annotations

from scholar_agent.config import Settings
from scholar_agent.indexes import BM25Index, DenseIndex
from scholar_agent.models import load_chunks

PER_QUERY_CANDIDATES = 8
PAPER_SEARCH_RESULTS = 4


class RetrievalEngine:
    """Load and call the lexical and semantic indexes."""

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

    def sparse_search(
        self,
        queries: list[str],
        top_k: int = PER_QUERY_CANDIDATES,
    ) -> list[dict]:
        return self.bm25.search(queries, top_k)

    def dense_search_many(
        self,
        queries: list[str],
        top_k: int = PER_QUERY_CANDIDATES,
    ) -> list[list[dict]]:
        return self.dense.search_many(queries, top_k)

    def search_within_paper(
        self,
        paper: str,
        query: str,
        top_k: int = PAPER_SEARCH_RESULTS,
    ) -> list[dict]:
        """Run hybrid retrieval over chunks from one paper."""
        indices = [
            index for index, item in enumerate(self.chunks) if item["paper"] == paper
        ]
        if not indices:
            return []
        chunks = [self.chunks[index] for index in indices]
        sparse = BM25Index(chunks).search([query], top_k)
        dense = self.dense.search_many([query], top_k, candidate_indices=indices)[0]
        return reciprocal_rank_fusion(sparse, dense)[:top_k]

    def expand_neighbors(self, chunk_id: str, radius: int = 1) -> list[dict]:
        """Return an inclusive chunk window within the seed paper."""
        seed = next((item for item in self.chunks if item["chunk_id"] == chunk_id), None)
        if seed is None:
            raise KeyError(f"Unknown chunk_id: {chunk_id}")
        return [
            item
            for item in self.chunks
            if item["paper"] == seed["paper"]
            and abs(item["chunk_index"] - seed["chunk_index"]) <= radius
        ]


def reciprocal_rank_fusion(*result_lists: list[dict], k: int = 60) -> list[dict]:
    """Fuse ranks by chunk ID; no weights, contribution models, or factories."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            items[chunk_id] = item
    chunk_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    return [{**items[chunk_id], "score": scores[chunk_id]} for chunk_id in chunk_ids]


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

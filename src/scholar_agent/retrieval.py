"""The direct retrieval path used by the Researcher agent."""

from __future__ import annotations

from pathlib import Path

from scholar_agent.config import Settings
from scholar_agent.graph_store import build_graph, graph_search, load_graph, save_graph
from scholar_agent.indexes import BM25Index, DenseIndex
from scholar_agent.models import load_chunks


class RetrievalEngine:
    """Load and call the three concrete retrievers without a tool registry."""

    def __init__(
        self,
        chunks: list[dict],
        bm25: BM25Index,
        dense: DenseIndex,
        graph: object,
        top_k: int = 20,
    ) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense
        self.graph = graph
        self.top_k = top_k

    @classmethod
    def load(cls, settings: Settings) -> RetrievalEngine:
        chunks = load_chunks(settings.chunks_path)
        bm25 = BM25Index.load(chunks, settings.index_dir / "bm25.json")
        dense = DenseIndex.load(chunks, settings.index_dir)
        graph = load_graph(settings.index_dir / "graph.json")
        return cls(chunks, bm25, dense, graph, settings.top_k)

    def sparse_search(self, queries: list[str]) -> list[dict]:
        return self.bm25.search(queries, self.top_k)

    def dense_search(self, queries: list[str]) -> list[dict]:
        return self.dense.search(queries, self.top_k)

    def dense_search_many(self, queries: list[str]) -> list[list[dict]]:
        return self.dense.search_many(queries, self.top_k)

    def graph_search(self, entities: list[str]) -> list[dict]:
        return graph_search(entities, self.graph, self.chunks, self.top_k)


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


def index_paths(data_dir: Path) -> tuple[Path, Path, Path]:
    index_dir = data_dir / "indexes"
    return index_dir / "bm25.json", index_dir / "dense.npy", index_dir / "graph.json"


def build_all_indexes(settings: Settings) -> dict[str, object]:
    """Build the three concrete indexes and return display-only summary values."""
    chunks = load_chunks(settings.chunks_path)
    bm25 = BM25Index(chunks)
    bm25.save(settings.index_dir / "bm25.json")

    dense = DenseIndex.build(chunks, settings.embedding_model)
    dense.save(settings.index_dir)

    graph = build_graph(chunks)
    save_graph(graph, settings.index_dir / "graph.json")
    return {
        "chunks": len(chunks),
        "entities": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "dense_backend": dense.backend,
    }

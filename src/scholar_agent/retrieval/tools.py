"""Typed retrieval tools used by agents and CLI."""

from __future__ import annotations

from typing import Any, Literal

from scholar_agent.graph.retrieve import GraphRetriever
from scholar_agent.models.corpus import Chunk, Paper
from scholar_agent.models.retrieval import (
    RankTrace,
    RetrievalFilters,
    RetrievalHit,
    RetrievalResult,
)
from scholar_agent.retrieval.chunk_store import ChunkStore
from scholar_agent.retrieval.dense import DenseIndex
from scholar_agent.retrieval.fusion import ranks_map, reciprocal_rank_fusion
from scholar_agent.retrieval.reranker import LexicalReranker, Reranker
from scholar_agent.retrieval.sparse import BM25Index


class RetrievalToolkit:
    """Independently testable retrieval façade over dense/sparse/graph indexes."""

    def __init__(
        self,
        store: ChunkStore,
        *,
        dense: DenseIndex | None = None,
        sparse: BM25Index | None = None,
        graph: GraphRetriever | None = None,
        reranker: Reranker | None = None,
        dense_top_k: int = 12,
        sparse_top_k: int = 12,
        fused_top_k: int = 20,
        rerank_top_k: int = 8,
        rrf_k: int = 60,
    ) -> None:
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.graph = graph
        self.reranker = reranker or LexicalReranker()
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.fused_top_k = fused_top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.store.get_chunk(chunk_id)

    def get_paper(self, paper_id: str) -> Paper | None:
        return self.store.get_paper(paper_id)

    def dense_search(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        if self.dense is None:
            raise RuntimeError("dense index is not loaded")
        top_k = k or self.dense_top_k
        hits = self.dense.search(query, k=top_k, filters=filters)
        traces = [
            RankTrace(chunk_id=h.chunk_id, dense_rank=h.dense_rank, final_rank=i)
            for i, h in enumerate(hits, start=1)
        ]
        return RetrievalResult(
            query=query,
            method="dense",
            hits=hits,
            traces=traces,
            debug={"k": top_k},
        )

    def sparse_search(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        if self.sparse is None:
            raise RuntimeError("sparse BM25 index is not loaded")
        top_k = k or self.sparse_top_k
        hits = self.sparse.search(query, k=top_k, filters=filters)
        traces = [
            RankTrace(chunk_id=h.chunk_id, sparse_rank=h.sparse_rank, final_rank=i)
            for i, h in enumerate(hits, start=1)
        ]
        return RetrievalResult(
            query=query,
            method="sparse",
            hits=hits,
            traces=traces,
            debug={"k": top_k},
        )

    def hybrid_search(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: RetrievalFilters | None = None,
        rerank: bool = False,
    ) -> RetrievalResult:
        if self.dense is None or self.sparse is None:
            raise RuntimeError("hybrid search requires both dense and sparse indexes")
        dense_hits = self.dense.search(query, k=self.dense_top_k, filters=filters)
        sparse_hits = self.sparse.search(query, k=self.sparse_top_k, filters=filters)
        dense_ids = [h.chunk_id for h in dense_hits]
        sparse_ids = [h.chunk_id for h in sparse_hits]
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self.rrf_k)
        fused = fused[: (k or self.fused_top_k)]

        dense_rank = ranks_map(dense_ids)
        sparse_rank = ranks_map(sparse_ids)
        by_id: dict[str, RetrievalHit] = {h.chunk_id: h for h in dense_hits}
        for h in sparse_hits:
            by_id.setdefault(h.chunk_id, h)

        hits: list[RetrievalHit] = []
        traces: list[RankTrace] = []
        for final_rank, (chunk_id, fused_score) in enumerate(fused, start=1):
            base = by_id.get(chunk_id) or self._hit_from_store(chunk_id, "hybrid")
            if base is None:
                continue
            hit = base.model_copy(
                update={
                    "dense_rank": dense_rank.get(chunk_id),
                    "sparse_rank": sparse_rank.get(chunk_id),
                    "fused_score": fused_score,
                    "score": fused_score,
                    "retrieval_method": "hybrid",
                }
            )
            hits.append(hit)
            traces.append(
                RankTrace(
                    chunk_id=chunk_id,
                    dense_rank=dense_rank.get(chunk_id),
                    sparse_rank=sparse_rank.get(chunk_id),
                    fused_score=fused_score,
                    final_rank=final_rank,
                )
            )

        method: Literal["hybrid", "hybrid_rerank"] = "hybrid"
        if rerank and hits:
            hits = self.reranker.rerank(query, hits, top_k=self.rerank_top_k)
            method = "hybrid_rerank"
            traces = [
                RankTrace(
                    chunk_id=h.chunk_id,
                    dense_rank=h.dense_rank,
                    sparse_rank=h.sparse_rank,
                    fused_score=h.fused_score,
                    rerank_score=h.rerank_score,
                    final_rank=i,
                )
                for i, h in enumerate(hits, start=1)
            ]

        return RetrievalResult(
            query=query,
            method=method,
            hits=hits,
            traces=traces,
            debug={
                "dense_top_k": self.dense_top_k,
                "sparse_top_k": self.sparse_top_k,
                "fused_top_k": k or self.fused_top_k,
                "rrf_k": self.rrf_k,
                "rerank": rerank,
                "dense_ids": dense_ids,
                "sparse_ids": sparse_ids,
            },
        )

    def graph_search(
        self,
        query: str,
        *,
        max_hops: int = 2,
        relation_filters: list[str] | None = None,
        k: int | None = None,
        exclude_chunk_ids: set[str] | None = None,
    ) -> RetrievalResult:
        if self.graph is None:
            raise RuntimeError(
                "graph index is not loaded; run `scholar-agent graph build` first"
            )
        return self.graph.search(
            query,
            max_hops=max_hops,
            relation_filters=relation_filters,
            k=k or self.rerank_top_k,
            exclude_chunk_ids=exclude_chunk_ids,
        )

    def search(
        self,
        query: str,
        *,
        mode: Literal["dense", "sparse", "hybrid", "hybrid_rerank", "graph"] = "hybrid_rerank",
        k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        if mode == "dense":
            return self.dense_search(query, k=k, filters=filters)
        if mode == "sparse":
            return self.sparse_search(query, k=k, filters=filters)
        if mode == "graph":
            return self.graph_search(query, k=k)
        if mode == "hybrid":
            return self.hybrid_search(query, k=k, filters=filters, rerank=False)
        return self.hybrid_search(query, k=k, filters=filters, rerank=True)

    def _hit_from_store(self, chunk_id: str, method: str) -> RetrievalHit | None:
        chunk = self.store.get_chunk(chunk_id)
        if chunk is None:
            return None
        return RetrievalHit(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section=chunk.section,
            retrieval_method=method,
        )

    def debug_dict(self, result: RetrievalResult) -> dict[str, Any]:
        return {
            "query": result.query,
            "method": result.method,
            "n_hits": len(result.hits),
            "hits": [
                {
                    "rank": i,
                    "chunk_id": h.chunk_id,
                    "paper_id": h.paper_id,
                    "pages": h.page_label(),
                    "section": h.section,
                    "score": h.score,
                    "dense_rank": h.dense_rank,
                    "sparse_rank": h.sparse_rank,
                    "fused_score": h.fused_score,
                    "rerank_score": h.rerank_score,
                    "snippet": h.snippet(160),
                }
                for i, h in enumerate(result.hits, start=1)
            ],
            "debug": result.debug,
        }

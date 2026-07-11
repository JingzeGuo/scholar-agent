"""Graph retrieval: entity linking → bounded paths → supporting chunks."""

from __future__ import annotations

import re
from typing import Any

from scholar_agent.graph.aliases import SEED_ALIASES
from scholar_agent.graph.store import KnowledgeGraphStore
from scholar_agent.ids import normalize_text
from scholar_agent.models.retrieval import RankTrace, RetrievalHit, RetrievalResult
from scholar_agent.retrieval.chunk_store import ChunkStore


class GraphRetriever:
    """Retrieve evidence chunks via schema-valid graph paths (≤ max_hops)."""

    def __init__(
        self,
        graph: KnowledgeGraphStore,
        chunks: ChunkStore,
        *,
        max_hops: int = 2,
    ) -> None:
        self.graph = graph
        self.chunks = chunks
        self.max_hops = max_hops
        # Build alias → entity_id map from graph nodes
        self._alias_index: dict[str, str] = {}
        for node_id, attrs in graph.graph.nodes(data=True):
            names = [str(attrs.get("canonical_name") or "")] + list(attrs.get("aliases") or [])
            for name in names:
                if name.strip():
                    self._alias_index[normalize_text(name)] = str(node_id)
        for surface, (canonical, _etype) in SEED_ALIASES.items():
            # map seed surfaces if canonical exists in graph
            for node_id, attrs in graph.graph.nodes(data=True):
                if normalize_text(str(attrs.get("canonical_name") or "")) == normalize_text(
                    canonical
                ):
                    self._alias_index[surface] = str(node_id)
                    break

    def link_entities(self, query: str) -> list[str]:
        """Link query spans to graph entity IDs (longest alias match)."""
        q = query.lower()
        hits: list[tuple[int, str]] = []
        for alias, eid in sorted(self._alias_index.items(), key=lambda x: -len(x[0])):
            if len(alias) < 2:
                continue
            if alias in q:
                # boundary-ish
                pattern = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", re.I)
                if pattern.search(query):
                    hits.append((len(alias), eid))
        # unique preserve longer-first
        seen: set[str] = set()
        ordered: list[str] = []
        for _, eid in sorted(hits, key=lambda x: -x[0]):
            if eid not in seen:
                seen.add(eid)
                ordered.append(eid)
        return ordered

    def search(
        self,
        query: str,
        *,
        max_hops: int | None = None,
        relation_filters: list[str] | None = None,
        k: int = 8,
        exclude_chunk_ids: set[str] | None = None,
    ) -> RetrievalResult:
        hops = max_hops if max_hops is not None else self.max_hops
        entity_ids = self.link_entities(query)
        paths = self.graph.paths_between(
            entity_ids,
            max_hops=hops,
            relation_filters=relation_filters,
            limit=100,
        )
        exclude = exclude_chunk_ids or set()

        # Score paths: confidence * length_penalty * query entity coverage
        scored_chunks: dict[str, tuple[float, dict[str, Any]]] = {}
        for path in paths:
            edges = path["edges"]
            if not edges:
                continue
            conf = 1.0
            for e in edges:
                conf *= float(e.get("confidence") or 0.5)
            length_penalty = 1.0 / (1.0 + 0.25 * (len(edges) - 1))
            path_score = conf * length_penalty
            for e in edges:
                chunk_id = str(e.get("chunk_id") or "")
                if not chunk_id or chunk_id in exclude:
                    continue
                prev = scored_chunks.get(chunk_id)
                if prev is None or path_score > prev[0]:
                    scored_chunks[chunk_id] = (path_score, e)

        ranked = sorted(scored_chunks.items(), key=lambda x: (-x[1][0], x[0]))[:k]
        hits: list[RetrievalHit] = []
        traces: list[RankTrace] = []
        for rank, (chunk_id, (score, _edge)) in enumerate(ranked, start=1):
            chunk = self.chunks.get_chunk(chunk_id)
            if chunk is None:
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    score=score,
                    retrieval_method="graph",
                )
            )
            traces.append(
                RankTrace(
                    chunk_id=chunk_id,
                    final_rank=rank,
                    fused_score=score,
                )
            )

        return RetrievalResult(
            query=query,
            method="graph",
            hits=hits,
            traces=traces,
            debug={
                "linked_entities": entity_ids,
                "n_paths": len(paths),
                "max_hops": hops,
                "relation_filters": relation_filters,
                "note": "Graph paths return supporting chunks, not standalone triples.",
            },
        )

"""Graph retrieval: entity linking → bounded paths → supporting chunks."""

from __future__ import annotations

import re
from typing import Any

from scholar_agent.graph.aliases import SEED_ALIASES
from scholar_agent.graph.store import KnowledgeGraphStore
from scholar_agent.ids import normalize_text
from scholar_agent.models.retrieval import RankTrace, RetrievalHit, RetrievalResult
from scholar_agent.retrieval.chunk_store import ChunkStore

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "what",
    "with",
}


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS}


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
        matches: list[tuple[int, int, int, str]] = []
        for alias, eid in sorted(self._alias_index.items(), key=lambda x: -len(x[0])):
            if len(alias) < 2:
                continue
            pattern = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", re.I)
            for match in pattern.finditer(query):
                matches.append((len(alias), match.start(), match.end(), eid))

        # Longest-span wins, preventing RAG from also matching inside Self-RAG.
        occupied: list[tuple[int, int]] = []
        seen: set[str] = set()
        ordered: list[str] = []
        for _length, start, end, eid in sorted(matches, key=lambda item: (-item[0], item[1])):
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            occupied.append((start, end))
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
        paths = self._query_paths(
            entity_ids,
            max_hops=hops,
            relation_filters=relation_filters,
        )
        exclude = exclude_chunk_ids or set()

        query_tokens = _tokens(query)
        scored_chunks: dict[str, tuple[float, dict[str, Any], dict[str, float]]] = {}
        for path in paths:
            edges = path["edges"]
            if not edges:
                continue
            length_penalty = 1.0 / (1.0 + 0.25 * (len(edges) - 1))
            edge_components = [self._score_edge(query_tokens, edge) for edge in edges]
            query_entity_coverage = len(set(path["nodes"]) & set(entity_ids)) / max(
                1, len(entity_ids)
            )
            path_score = (
                sum(component["combined"] for component in edge_components)
                / len(edge_components)
                * length_penalty
                * (0.7 + 0.3 * query_entity_coverage)
            )
            for e, component in zip(edges, edge_components, strict=True):
                chunk_id = str(e.get("chunk_id") or "")
                if not chunk_id or chunk_id in exclude:
                    continue
                score = (0.7 * component["combined"] + 0.3 * path_score) * length_penalty
                prev = scored_chunks.get(chunk_id)
                if prev is None or score > prev[0]:
                    scored_chunks[chunk_id] = (score, e, component)

        max_relevance = max(
            (value[2]["query_relevance"] for value in scored_chunks.values()),
            default=0.0,
        )
        min_relevance = max_relevance * 0.75 if max_relevance > 0 else 0.0
        rankable = [
            item for item in scored_chunks.items() if item[1][2]["query_relevance"] >= min_relevance
        ]
        ranked = sorted(rankable, key=lambda x: (-x[1][0], x[0]))[:k]
        hits: list[RetrievalHit] = []
        traces: list[RankTrace] = []
        score_debug: list[dict[str, Any]] = []
        for rank, (chunk_id, (score, _edge, component)) in enumerate(ranked, start=1):
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
            score_debug.append({"chunk_id": chunk_id, **component, "final_score": score})

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
                "min_query_relevance": min_relevance,
                "score_components": score_debug,
                "note": "Graph paths return supporting chunks, not standalone triples.",
            },
        )

    def _query_paths(
        self,
        entity_ids: list[str],
        *,
        max_hops: int,
        relation_filters: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Prioritize direct paths between query entities, then bounded neighborhoods."""
        allowed = set(relation_filters) if relation_filters else None
        paths: list[dict[str, Any]] = []
        query_set = set(entity_ids)
        for source in entity_ids:
            for _u, target, _key, attrs in self.graph.graph.out_edges(
                source,
                keys=True,
                data=True,
            ):
                relation_type = str(attrs.get("relation_type") or "")
                if target not in query_set or target == source:
                    continue
                if allowed is not None and relation_type not in allowed:
                    continue
                paths.append(
                    {
                        "nodes": [source, str(target)],
                        "edges": [self._edge_dict(source, str(target), attrs)],
                        "hops": 1,
                    }
                )

        for source in entity_ids:
            paths.extend(
                self.graph.paths_between(
                    [source],
                    max_hops=max_hops,
                    relation_filters=relation_filters,
                    limit=50,
                )
            )

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for path in paths:
            key = tuple(str(edge.get("relation_id") or "") for edge in path["edges"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
        return deduped[:200]

    @staticmethod
    def _edge_dict(source: str, target: str, attrs: dict[str, Any]) -> dict[str, Any]:
        return {
            "relation_id": attrs.get("relation_id"),
            "relation_type": attrs.get("relation_type"),
            "chunk_id": attrs.get("chunk_id"),
            "paper_id": attrs.get("paper_id"),
            "page_number": attrs.get("page_number"),
            "page_end": attrs.get("page_end") or attrs.get("page_number"),
            "evidence_span": attrs.get("evidence_span"),
            "confidence": attrs.get("confidence"),
            "subject": source,
            "object": target,
        }

    def _score_edge(self, query_tokens: set[str], edge: dict[str, Any]) -> dict[str, float]:
        subject = str(edge.get("subject") or "")
        object_ = str(edge.get("object") or "")
        subject_name = str(self.graph.graph.nodes.get(subject, {}).get("canonical_name") or "")
        object_name = str(self.graph.graph.nodes.get(object_, {}).get("canonical_name") or "")
        evidence = str(edge.get("evidence_span") or "")
        relation_type = str(edge.get("relation_type") or "").replace("_", " ")
        searchable = _tokens(" ".join((subject_name, object_name, evidence, relation_type)))
        relevance = len(query_tokens & searchable) / max(1, len(query_tokens))

        confidence = min(1.0, max(0.0, float(edge.get("confidence") or 0.0)))
        chunk_id = str(edge.get("chunk_id") or "")
        chunk = self.chunks.get_chunk(chunk_id)
        span_tokens = len(_TOKEN_RE.findall(evidence))
        span_quality = min(1.0, span_tokens / 12.0) if evidence.strip() else 0.0
        provenance_quality = 1.0 if chunk is not None else 0.0
        evidence_quality = 0.65 * span_quality + 0.35 * provenance_quality
        combined = 0.55 * relevance + 0.30 * confidence + 0.15 * evidence_quality
        return {
            "query_relevance": relevance,
            "relation_confidence": confidence,
            "evidence_quality": evidence_quality,
            "combined": combined,
        }

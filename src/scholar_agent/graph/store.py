"""NetworkX MultiDiGraph store with node-link JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from scholar_agent.models.graph import Entity, EntityType, Relation, RelationType


class KnowledgeGraphStore:
    """Evidence-linked knowledge graph over NetworkX MultiDiGraph."""

    def __init__(self, graph: nx.MultiDiGraph | None = None) -> None:
        self.graph: nx.MultiDiGraph = graph if graph is not None else nx.MultiDiGraph()

    @classmethod
    def from_entities_relations(
        cls,
        entities: list[Entity],
        relations: list[Relation],
    ) -> KnowledgeGraphStore:
        g = nx.MultiDiGraph()
        for ent in entities:
            g.add_node(
                ent.entity_id,
                entity_id=ent.entity_id,
                entity_type=ent.entity_type.value,
                canonical_name=ent.canonical_name,
                aliases=list(ent.aliases),
                description=ent.description,
            )
        for rel in relations:
            if not rel.subject_entity_id or not rel.object_entity_id:
                continue
            if rel.subject_entity_id not in g:
                continue
            if rel.object_entity_id not in g:
                continue
            # MultiDiGraph edge key = relation_id for stability
            g.add_edge(
                rel.subject_entity_id,
                rel.object_entity_id,
                key=rel.relation_id,
                relation_id=rel.relation_id,
                relation_type=rel.relation_type.value,
                subject_surface=rel.subject_surface,
                object_surface=rel.object_surface,
                evidence_span=rel.evidence_span,
                paper_id=rel.paper_id,
                chunk_id=rel.chunk_id,
                page_number=rel.page_number,
                confidence=rel.confidence,
            )
        return cls(g)

    def save_node_link_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(self.graph, edges="links")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load_node_link_json(cls, path: Path | str) -> KnowledgeGraphStore:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        # networkx 3.x uses edges="links" for node-link
        try:
            g = json_graph.node_link_graph(data, edges="links", directed=True, multigraph=True)
        except TypeError:
            g = json_graph.node_link_graph(data, directed=True, multigraph=True)
        if not isinstance(g, nx.MultiDiGraph):
            g = nx.MultiDiGraph(g)
        return cls(g)

    def entities(self) -> list[Entity]:
        out: list[Entity] = []
        for node_id, attrs in self.graph.nodes(data=True):
            out.append(
                Entity(
                    entity_id=str(attrs.get("entity_id") or node_id),
                    entity_type=EntityType(attrs.get("entity_type") or EntityType.METHOD),
                    canonical_name=str(attrs.get("canonical_name") or node_id),
                    aliases=list(attrs.get("aliases") or []),
                    description=attrs.get("description"),
                )
            )
        return out

    def relations(self) -> list[Relation]:
        out: list[Relation] = []
        for u, v, _key, attrs in self.graph.edges(keys=True, data=True):
            evidence = str(attrs.get("evidence_span") or "")
            if not evidence.strip():
                continue
            out.append(
                Relation(
                    relation_id=str(attrs.get("relation_id") or _key),
                    subject_surface=str(attrs.get("subject_surface") or u),
                    object_surface=str(attrs.get("object_surface") or v),
                    subject_entity_id=str(u),
                    object_entity_id=str(v),
                    subject_type=EntityType(self.graph.nodes[u].get("entity_type"))
                    if u in self.graph.nodes
                    else None,
                    object_type=EntityType(self.graph.nodes[v].get("entity_type"))
                    if v in self.graph.nodes
                    else None,
                    relation_type=RelationType(attrs.get("relation_type") or RelationType.USES),
                    evidence_span=evidence,
                    paper_id=str(attrs.get("paper_id") or ""),
                    chunk_id=str(attrs.get("chunk_id") or ""),
                    page_number=int(attrs.get("page_number") or 1),
                    confidence=float(attrs.get("confidence") or 0.0),
                )
            )
        return out

    def number_of_nodes(self) -> int:
        return int(self.graph.number_of_nodes())

    def number_of_edges(self) -> int:
        return int(self.graph.number_of_edges())

    def isolated_nodes(self) -> list[str]:
        return [str(n) for n in nx.isolates(self.graph)]

    def neighbors(self, entity_id: str, *, max_hops: int = 1) -> set[str]:
        if entity_id not in self.graph:
            return set()
        if max_hops <= 1:
            return set(self.graph.successors(entity_id)) | set(self.graph.predecessors(entity_id))
        # BFS undirected view up to max_hops
        seen = {entity_id}
        frontier = {entity_id}
        for _ in range(max_hops):
            nxt: set[str] = set()
            for node in frontier:
                nxt |= set(self.graph.successors(node)) | set(self.graph.predecessors(node))
            nxt -= seen
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
        seen.discard(entity_id)
        return seen

    def paths_between(
        self,
        sources: list[str],
        *,
        max_hops: int = 2,
        relation_filters: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return schema-valid paths of length 1..max_hops starting from sources."""
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        allowed = set(relation_filters) if relation_filters else None
        results: list[dict[str, Any]] = []

        for source in sources:
            if source not in self.graph:
                continue
            # DFS limited depth
            stack: list[tuple[str, list[str], list[dict[str, Any]]]] = [
                (source, [source], [])
            ]
            while stack and len(results) < limit:
                node, path_nodes, path_edges = stack.pop()
                if len(path_edges) >= 1:
                    results.append(
                        {
                            "nodes": list(path_nodes),
                            "edges": list(path_edges),
                            "hops": len(path_edges),
                        }
                    )
                    if len(results) >= limit:
                        break
                if len(path_edges) >= max_hops:
                    continue
                # outgoing
                for _, nbr, _key, attrs in self.graph.out_edges(node, keys=True, data=True):
                    rel_type = str(attrs.get("relation_type") or "")
                    if allowed is not None and rel_type not in allowed:
                        continue
                    if nbr in path_nodes:
                        continue
                    edge = {
                        "relation_id": attrs.get("relation_id"),
                        "relation_type": rel_type,
                        "chunk_id": attrs.get("chunk_id"),
                        "paper_id": attrs.get("paper_id"),
                        "page_number": attrs.get("page_number"),
                        "evidence_span": attrs.get("evidence_span"),
                        "confidence": attrs.get("confidence"),
                        "subject": node,
                        "object": nbr,
                    }
                    stack.append((nbr, path_nodes + [nbr], path_edges + [edge]))
                # incoming (treat as reverse traversal for discovery)
                for pred, _, _key, attrs in self.graph.in_edges(node, keys=True, data=True):
                    rel_type = str(attrs.get("relation_type") or "")
                    if allowed is not None and rel_type not in allowed:
                        continue
                    if pred in path_nodes:
                        continue
                    edge = {
                        "relation_id": attrs.get("relation_id"),
                        "relation_type": rel_type,
                        "chunk_id": attrs.get("chunk_id"),
                        "paper_id": attrs.get("paper_id"),
                        "page_number": attrs.get("page_number"),
                        "evidence_span": attrs.get("evidence_span"),
                        "confidence": attrs.get("confidence"),
                        "subject": pred,
                        "object": node,
                    }
                    stack.append((pred, path_nodes + [pred], path_edges + [edge]))
        return results[:limit]

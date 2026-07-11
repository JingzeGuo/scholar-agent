"""Graph statistics for inspection and portfolio reporting."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from scholar_agent.graph.store import KnowledgeGraphStore


class GraphStats(BaseModel):
    n_nodes: int = 0
    n_edges: int = 0
    n_isolated_nodes: int = 0
    isolated_node_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    node_type_counts: dict[str, int] = Field(default_factory=dict)
    relation_type_counts: dict[str, int] = Field(default_factory=dict)
    n_relations_with_evidence: int = 0
    n_relations_missing_evidence: int = 0
    mean_degree: float = 0.0
    notes: list[str] = Field(default_factory=list)


def compute_graph_stats(store: KnowledgeGraphStore) -> GraphStats:
    g = store.graph
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    isolated = store.isolated_nodes()
    node_types = Counter(
        str(attrs.get("entity_type") or "Unknown") for _, attrs in g.nodes(data=True)
    )
    rel_types = Counter(
        str(attrs.get("relation_type") or "Unknown")
        for _, _, _, attrs in g.edges(keys=True, data=True)
    )
    with_ev = 0
    missing_ev = 0
    for _, _, _, attrs in g.edges(keys=True, data=True):
        span = str(attrs.get("evidence_span") or "").strip()
        if span and attrs.get("chunk_id") and attrs.get("paper_id"):
            with_ev += 1
        else:
            missing_ev += 1

    degrees = [d for _, d in g.degree()]
    mean_degree = (sum(degrees) / len(degrees)) if degrees else 0.0
    rate = (len(isolated) / n_nodes) if n_nodes else 0.0

    notes = [
        "Graph edges are not independent facts; always join supporting chunks.",
        "Isolated nodes often come from alias seeds never linked by extraction.",
    ]
    return GraphStats(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_isolated_nodes=len(isolated),
        isolated_node_rate=rate,
        node_type_counts=dict(node_types),
        relation_type_counts=dict(rel_types),
        n_relations_with_evidence=with_ev,
        n_relations_missing_evidence=missing_ev,
        mean_degree=mean_degree,
        notes=notes,
    )

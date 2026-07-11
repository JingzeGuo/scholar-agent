"""Evidence-linked knowledge graph: extract, resolve, store, retrieve."""

from scholar_agent.graph.pipeline import GraphBuildResult, build_knowledge_graph
from scholar_agent.graph.retrieve import GraphRetriever
from scholar_agent.graph.stats import GraphStats, compute_graph_stats
from scholar_agent.graph.store import KnowledgeGraphStore

__all__ = [
    "GraphBuildResult",
    "GraphRetriever",
    "GraphStats",
    "KnowledgeGraphStore",
    "build_knowledge_graph",
    "compute_graph_stats",
]

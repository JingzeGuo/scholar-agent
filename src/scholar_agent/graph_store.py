"""Lightweight entity co-occurrence GraphRAG."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx

ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]+)+|[A-Z]{2,10}|"
    r"[A-Z][a-z]+(?:\s+[A-Z][A-Za-z-]+){1,3})\b",
)
STOP_ENTITIES = {
    "abstract",
    "introduction",
    "related work",
    "large language models",
    "retrieval augmented generation",
}


def normalize_entity(name: str) -> str:
    return " ".join(name.casefold().strip().split())


def _word_form(name: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", normalize_entity(name)))


def extract_entities(text: str, limit: int = 20) -> list[str]:
    """Extract normalized acronym, hyphenated-method, and title-case entities."""
    entities: list[str] = []
    seen: set[str] = set()
    for match in ENTITY_RE.finditer(text):
        entity = normalize_entity(match.group(0).strip(" ,.;:()[]"))
        if entity in STOP_ENTITIES or entity in seen:
            continue
        seen.add(entity)
        entities.append(entity)
        if len(entities) >= limit:
            break
    return entities


def build_graph(chunks: list[dict]) -> nx.Graph:
    """Connect entities appearing in the same chunk and retain chunk IDs."""
    graph = nx.Graph()
    for chunk in chunks:
        entities = extract_entities(chunk["text"])
        for entity in entities:
            if entity not in graph:
                graph.add_node(entity, chunks=[])
            node_chunks: list[str] = graph.nodes[entity]["chunks"]
            if chunk["chunk_id"] not in node_chunks:
                node_chunks.append(chunk["chunk_id"])
        for left, right in combinations(entities, 2):
            if graph.has_edge(left, right):
                graph[left][right]["weight"] += 1
            else:
                graph.add_edge(left, right, weight=1)
    return graph


def save_graph(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nx.node_link_data(graph), ensure_ascii=False), encoding="utf-8")


def load_graph(path: Path) -> nx.Graph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return nx.node_link_graph(payload)


def _matching_nodes(graph: nx.Graph, entity: str) -> list[str]:
    key = normalize_entity(entity)
    word_form = _word_form(key)
    exact = [
        node for node in graph if normalize_entity(node) == key or _word_form(node) == word_form
    ]
    if exact:
        return exact

    if len(word_form.split()) > 1:
        normalized = [node for node in graph if _word_form(node) == word_form]
        if normalized:
            return normalized

    compact = word_form.replace(" ", "")
    if len(compact) < 5:
        return []
    return [
        node
        for node in graph
        if compact in _word_form(node).replace(" ", "")
        or _word_form(node).replace(" ", "") in compact
    ]


def graph_search(
    entities: list[str],
    graph: nx.Graph,
    chunks: list[dict],
    top_k: int = 20,
) -> list[dict]:
    """Retrieve chunks through matching entity nodes and one-hop neighbors."""
    chunk_scores: defaultdict[str, float] = defaultdict(float)
    for entity in entities:
        for node in _matching_nodes(graph, entity):
            for chunk_id in graph.nodes[node].get("chunks", []):
                chunk_scores[chunk_id] += 1.0
            for neighbor in graph.neighbors(node):
                for chunk_id in graph.nodes[neighbor].get("chunks", []):
                    chunk_scores[chunk_id] += 0.5

    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    ranked = sorted(chunk_scores, key=lambda item: (-chunk_scores[item], item))[:top_k]
    return [
        {**by_id[chunk_id], "score": float(chunk_scores[chunk_id])}
        for chunk_id in ranked
        if chunk_id in by_id
    ]

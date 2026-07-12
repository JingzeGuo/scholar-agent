"""Build the evidence-linked knowledge graph from the canonical chunk store."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scholar_agent.config import AppConfig, load_config
from scholar_agent.graph.evidence import validate_relations
from scholar_agent.graph.extract import (
    EXTRACTION_CACHE_SCHEMA,
    extract_from_chunk,
    extract_paper_structure,
)
from scholar_agent.graph.resolver import EntityResolver, LLMEntityDisambiguator
from scholar_agent.graph.stats import GraphStats, compute_graph_stats
from scholar_agent.graph.store import KnowledgeGraphStore
from scholar_agent.ids import make_relation_id
from scholar_agent.llm.client import create_llm_client
from scholar_agent.logging import get_logger
from scholar_agent.models.corpus import Chunk, Paper
from scholar_agent.models.graph import Entity, EntityType, Relation
from scholar_agent.retrieval.chunk_store import ChunkStore
from scholar_agent.storage.cache import DiskCache
from scholar_agent.storage.jsonl import JsonlRepository

logger = get_logger(__name__)


@dataclass
class GraphBuildResult:
    store: KnowledgeGraphStore
    entities: list[Entity]
    relations: list[Relation]
    stats: GraphStats
    entities_path: Path
    relations_path: Path
    graph_path: Path
    stats_path: Path


def build_knowledge_graph(
    *,
    config: AppConfig | None = None,
    processed_dir: Path | str | None = None,
    limit_chunks: int | None = None,
    force: bool = False,
    resolver: EntityResolver | None = None,
    use_llm_resolution: bool = False,
    max_llm_resolutions: int = 50,
) -> GraphBuildResult:
    """Extract → resolve → validate → persist graph artifacts."""
    cfg = config or load_config()
    processed = Path(processed_dir or cfg.paths.processed_dir)
    chunk_store = ChunkStore.from_processed_dir(processed)

    entities_path = processed / "entities.jsonl"
    relations_path = processed / "relations.jsonl"
    graph_path = processed / "knowledge_graph.json"
    stats_path = processed / "graph_stats.json"

    if not force and entities_path.is_file() and relations_path.is_file() and graph_path.is_file():
        logger.info("loading existing knowledge graph from %s", graph_path)
        store = KnowledgeGraphStore.load_node_link_json(graph_path)
        entities = JsonlRepository(entities_path, Entity).read_all()
        relations = JsonlRepository(relations_path, Relation).read_all()
        stats = compute_graph_stats(store)
        stats_path.write_text(
            json.dumps(stats.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )
        return GraphBuildResult(
            store=store,
            entities=entities,
            relations=relations,
            stats=stats,
            entities_path=entities_path,
            relations_path=relations_path,
            graph_path=graph_path,
            stats_path=stats_path,
        )

    chunks = chunk_store.chunks
    if limit_chunks is not None:
        chunks = chunks[:limit_chunks]

    papers_by_id = chunk_store.by_paper_id
    chunks_by_paper: dict[str, list[Chunk]] = {}
    for ch in chunks:
        chunks_by_paper.setdefault(ch.paper_id, []).append(ch)

    if resolver is None:
        disambiguator = None
        if use_llm_resolution:
            disambiguator = LLMEntityDisambiguator(
                create_llm_client(cfg),
                max_calls=max_llm_resolutions,
            )
        resolver = EntityResolver(disambiguator=disambiguator)
    raw_relations: list[Relation] = []

    # Paper structural edges
    for paper_id, paper_chunks in chunks_by_paper.items():
        paper = papers_by_id.get(paper_id)
        if paper is None:
            # Minimal paper stub
            paper = Paper(
                paper_id=paper_id,
                title=paper_id,
                pdf_path="unknown.pdf",
                content_hash="0" * 16,
            )
        # Ensure paper entity exists
        resolver.register_surface(paper.title, EntityType.PAPER, preferred_canonical=paper.title)
        raw_relations.extend(extract_paper_structure(paper, paper_chunks))

    # Chunk-level extraction (optional disk cache under processed/.cache/extraction)
    extraction_cache = DiskCache(
        root=processed / ".cache",
        namespace="extraction",
        schema_version=EXTRACTION_CACHE_SCHEMA,
    )
    for i, chunk in enumerate(chunks):
        if i and i % 500 == 0:
            logger.info(
                "extracted relations from %s/%s chunks cache=%s",
                i,
                len(chunks),
                extraction_cache.stats.as_dict(),
            )
        raw_relations.extend(extract_from_chunk(chunk, cache=extraction_cache))
    logger.info("extraction_cache_stats %s", extraction_cache.stats.as_dict())

    logger.info("raw relations before validation: %s", len(raw_relations))
    grounded = validate_relations(raw_relations, chunk_store.by_chunk_id)
    logger.info("relations with grounded evidence: %s", len(grounded))

    # Resolve entities on grounded relations
    final_relations: list[Relation] = []
    for rel in grounded:
        sub_type = rel.subject_type or EntityType.METHOD
        obj_type = rel.object_type or EntityType.METHOD
        sub = resolver.register_surface(rel.subject_surface, sub_type)
        obj = resolver.register_surface(rel.object_surface, obj_type)
        if sub.entity_id == obj.entity_id:
            continue
        rid = make_relation_id(
            subject_entity_id=sub.entity_id,
            relation_type=rel.relation_type.value,
            object_entity_id=obj.entity_id,
            chunk_id=rel.chunk_id,
            evidence_span=rel.evidence_span,
        )
        final_relations.append(
            rel.model_copy(
                update={
                    "relation_id": rid,
                    "subject_entity_id": sub.entity_id,
                    "object_entity_id": obj.entity_id,
                    "subject_type": sub.entity_type,
                    "object_type": obj.entity_type,
                }
            )
        )

    # Dedupe by relation_id
    by_id = {r.relation_id: r for r in final_relations}
    final_relations = list(by_id.values())
    entities = resolver.all_entities()

    store = KnowledgeGraphStore.from_entities_relations(entities, final_relations)
    stats = compute_graph_stats(store)

    # Persist
    JsonlRepository(entities_path, Entity).write_all(sorted(entities, key=lambda e: e.entity_id))
    JsonlRepository(relations_path, Relation).write_all(
        sorted(final_relations, key=lambda r: r.relation_id)
    )
    store.save_node_link_json(graph_path)
    stats_path.write_text(
        json.dumps(stats.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )

    # Optional: resolution decisions for audit
    decisions_path = processed / "entity_resolution_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as handle:
        for d in resolver.decisions:
            handle.write(
                json.dumps(
                    {
                        "surface": d.surface,
                        "entity_type": d.entity_type.value,
                        "canonical_name": d.canonical_name,
                        "entity_id": d.entity_id,
                        "method": d.method,
                        "score": d.score,
                        "candidates": d.candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info(
        "graph built: nodes=%s edges=%s isolated_rate=%.3f",
        stats.n_nodes,
        stats.n_edges,
        stats.isolated_node_rate,
    )
    return GraphBuildResult(
        store=store,
        entities=entities,
        relations=final_relations,
        stats=stats,
        entities_path=entities_path,
        relations_path=relations_path,
        graph_path=graph_path,
        stats_path=stats_path,
    )

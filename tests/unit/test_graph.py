"""Phase 4 knowledge-graph tests."""

from __future__ import annotations

import json
from pathlib import Path

from scholar_agent.graph.evidence import find_evidence_span, validate_relation_against_chunk
from scholar_agent.graph.extract import extract_from_chunk
from scholar_agent.graph.pipeline import build_knowledge_graph
from scholar_agent.graph.resolver import EntityResolver
from scholar_agent.graph.retrieve import GraphRetriever
from scholar_agent.graph.stats import compute_graph_stats
from scholar_agent.graph.store import KnowledgeGraphStore
from scholar_agent.ids import content_hash, make_chunk_id, make_entity_id, make_relation_id
from scholar_agent.models.corpus import Chunk, Paper
from scholar_agent.models.graph import Entity, EntityType, Relation, RelationType
from scholar_agent.retrieval.chunk_store import ChunkStore
from scholar_agent.storage.jsonl import JsonlRepository


def _chunk(paper_id: str, text: str, page: int = 1, section: str | None = "Method") -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(paper_id, page_start=page, page_end=page, text=text, section=section),
        paper_id=paper_id,
        text=text,
        page_start=page,
        page_end=page,
        section=section,
        token_count=len(text.split()),
        content_hash=content_hash(text),
    )


def test_evidence_span_must_ground_in_chunk() -> None:
    chunk = _chunk(
        "paper_x",
        "Self-RAG retrieves passages on demand and critiques generation.",
    )
    ok_span = find_evidence_span(chunk.text, "Self-RAG retrieves passages on demand")
    assert ok_span is not None
    bad = find_evidence_span(chunk.text, "This sentence is not in the chunk at all.")
    assert bad is None

    rel = Relation(
        relation_id="rel_tmp",
        subject_surface="Self-RAG",
        object_surface="passages",
        relation_type=RelationType.USES,
        evidence_span="Self-RAG retrieves passages on demand",
        paper_id=chunk.paper_id,
        chunk_id=chunk.chunk_id,
        page_number=1,
        confidence=0.7,
    )
    assert validate_relation_against_chunk(rel, chunk) is not None
    bad_rel = rel.model_copy(update={"evidence_span": "totally fabricated evidence span xyz"})
    assert validate_relation_against_chunk(bad_rel, chunk) is None


def test_entity_alias_fixtures_resolve(repo_root: Path) -> None:
    path = repo_root / "tests" / "fixtures" / "entity_aliases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    resolver = EntityResolver()
    for row in rows:
        ent = resolver.register_surface(row["surface"], EntityType(row["entity_type"]))
        assert ent.canonical_name == row["canonical_name"], (
            f"{row['surface']} -> {ent.canonical_name} != {row['canonical_name']}"
        )
    # Same canonical for Self-RAG variants
    a = resolver.register_surface("Self-RAG", EntityType.METHOD)
    b = resolver.register_surface("self rag", EntityType.METHOD)
    assert a.entity_id == b.entity_id


def test_string_similarity_merges_near_duplicates() -> None:
    resolver = EntityResolver()
    e1 = resolver.register_surface("Dense Passage Retrieval", EntityType.METHOD)
    # High similarity to existing canonical
    e2 = resolver.register_surface("dense passage retrieval", EntityType.METHOD)
    assert e1.entity_id == e2.entity_id


def test_extract_and_build_graph_with_supporting_chunks(tmp_path: Path) -> None:
    chunks = [
        _chunk(
            "paper_self_rag",
            "Self-RAG: Learning to Retrieve. Self-RAG proposes a retrieve-on-demand framework "
            "and uses reflection tokens. Self-RAG evaluates on Natural Questions and HotpotQA.",
            page=3,
        ),
        _chunk(
            "paper_crag",
            "Corrective RAG evaluates retrieved documents. CRAG uses BM25 and dense retrieval "
            "and outperforms standard RAG on MS MARCO.",
            page=2,
        ),
        _chunk(
            "paper_dpr",
            "Dense Passage Retrieval uses dual-encoders for open-domain question answering "
            "over Wikipedia. DPR reports Exact Match gains on Natural Questions.",
            page=1,
        ),
    ]
    papers = [
        Paper(
            paper_id="paper_self_rag",
            title="Self-RAG: Learning to Retrieve",
            authors=["Akari Asai"],
            pdf_path="self.pdf",
            content_hash=content_hash("self"),
        ),
        Paper(
            paper_id="paper_crag",
            title="Corrective Retrieval Augmented Generation",
            authors=["Shi-Qi Yan"],
            pdf_path="crag.pdf",
            content_hash=content_hash("crag"),
        ),
        Paper(
            paper_id="paper_dpr",
            title="Dense Passage Retrieval for Open-Domain Question Answering",
            authors=["Vladimir Karpukhin"],
            pdf_path="dpr.pdf",
            content_hash=content_hash("dpr"),
        ),
    ]
    processed = tmp_path / "processed"
    processed.mkdir()
    JsonlRepository(processed / "chunks.jsonl", Chunk).write_all(chunks)
    JsonlRepository(processed / "papers.jsonl", Paper).write_all(papers)

    result = build_knowledge_graph(processed_dir=processed, force=True)
    assert result.stats.n_nodes > 0
    assert result.stats.n_edges > 0
    # Every persisted relation must have evidence
    assert result.stats.n_relations_missing_evidence == 0
    assert all(r.evidence_span.strip() and r.chunk_id for r in result.relations)

    # Round-trip node-link JSON
    loaded = KnowledgeGraphStore.load_node_link_json(result.graph_path)
    assert loaded.number_of_edges() == result.store.number_of_edges()

    # Graph retrieval returns supporting chunks with pages
    store = ChunkStore.from_processed_dir(processed)
    retriever = GraphRetriever(result.store, store, max_hops=2)
    hits = retriever.search("Self-RAG Natural Questions", k=5)
    assert hits.method == "graph"
    if hits.hits:
        assert all(h.chunk_id in store.by_chunk_id for h in hits.hits)
        assert all(h.page_start >= 1 for h in hits.hits)
        assert all(h.retrieval_method == "graph" for h in hits.hits)


def test_graph_paths_not_unfiltered_expansion() -> None:
    e_method = Entity(
        entity_id=make_entity_id("Method", "Self-RAG"),
        entity_type=EntityType.METHOD,
        canonical_name="Self-RAG",
    )
    e_data = Entity(
        entity_id=make_entity_id("Dataset", "HotpotQA"),
        entity_type=EntityType.DATASET,
        canonical_name="HotpotQA",
    )
    e_noise = Entity(
        entity_id=make_entity_id("Method", "Unrelated"),
        entity_type=EntityType.METHOD,
        canonical_name="Unrelated",
    )
    rel = Relation(
        relation_id=make_relation_id(
            subject_entity_id=e_method.entity_id,
            relation_type="EVALUATES_ON",
            object_entity_id=e_data.entity_id,
            chunk_id="chunk_1",
            evidence_span="Self-RAG evaluates on HotpotQA",
        ),
        subject_surface="Self-RAG",
        object_surface="HotpotQA",
        subject_entity_id=e_method.entity_id,
        object_entity_id=e_data.entity_id,
        subject_type=EntityType.METHOD,
        object_type=EntityType.DATASET,
        relation_type=RelationType.EVALUATES_ON,
        evidence_span="Self-RAG evaluates on HotpotQA",
        paper_id="p1",
        chunk_id="chunk_1",
        page_number=3,
        confidence=0.9,
    )
    store = KnowledgeGraphStore.from_entities_relations([e_method, e_data, e_noise], [rel])
    paths = store.paths_between([e_method.entity_id], max_hops=1, limit=20)
    assert paths
    # Isolated noise node should not appear in 1-hop paths from Self-RAG
    for path in paths:
        assert e_noise.entity_id not in path["nodes"]


def test_stats_report_isolated_rate() -> None:
    entities = [
        Entity(
            entity_id=make_entity_id("Method", "A"),
            entity_type=EntityType.METHOD,
            canonical_name="A",
        ),
        Entity(
            entity_id=make_entity_id("Method", "B"),
            entity_type=EntityType.METHOD,
            canonical_name="B",
        ),
        Entity(
            entity_id=make_entity_id("Method", "Isolated"),
            entity_type=EntityType.METHOD,
            canonical_name="Isolated",
        ),
    ]
    rel = Relation(
        relation_id="rel_ab",
        subject_surface="A",
        object_surface="B",
        subject_entity_id=entities[0].entity_id,
        object_entity_id=entities[1].entity_id,
        relation_type=RelationType.COMPARES_WITH,
        evidence_span="A compares with B in experiments",
        paper_id="p",
        chunk_id="c",
        page_number=1,
        confidence=0.5,
    )
    store = KnowledgeGraphStore.from_entities_relations(entities, [rel])
    stats = compute_graph_stats(store)
    assert stats.n_nodes == 3
    assert stats.n_edges == 1
    assert stats.n_isolated_nodes == 1
    assert abs(stats.isolated_node_rate - 1 / 3) < 1e-6
    assert stats.n_relations_with_evidence == 1


def test_extractor_emits_grounded_relations() -> None:
    chunk = _chunk(
        "paper_x",
        "Self-RAG proposes a retrieve-on-demand framework. Self-RAG evaluates on HotpotQA.",
    )
    rels = extract_from_chunk(chunk)
    assert rels
    for rel in rels:
        assert rel.evidence_span
        assert find_evidence_span(chunk.text, rel.evidence_span) is not None

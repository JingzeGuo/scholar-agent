"""Phase 4 knowledge-graph tests."""

from __future__ import annotations

import json
from pathlib import Path

from scholar_agent.graph.evidence import (
    find_evidence_span,
    localize_relation_to_pages,
    locate_evidence_pages,
    validate_relation_against_chunk,
)
from scholar_agent.graph.extract import extract_from_chunk
from scholar_agent.graph.pipeline import build_knowledge_graph, validate_graph_build_meta
from scholar_agent.graph.resolver import (
    EntityResolutionJudgment,
    EntityResolver,
    LLMEntityDisambiguator,
    ResolutionCandidate,
)
from scholar_agent.graph.retrieve import GraphRetriever
from scholar_agent.graph.stats import compute_graph_stats
from scholar_agent.graph.store import KnowledgeGraphStore
from scholar_agent.ids import content_hash, make_chunk_id, make_entity_id, make_relation_id
from scholar_agent.llm.client import ChatResponse
from scholar_agent.models.corpus import Chunk, Paper, PaperPage
from scholar_agent.models.graph import Entity, EntityType, Relation, RelationType
from scholar_agent.retrieval.chunk_store import ChunkStore
from scholar_agent.storage.jsonl import JsonlRepository


class SemanticTestEmbedder:
    model_name = "semantic-test"
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        low = text.lower()
        if "alpha" in low or "semantic passage" in low:
            return [1.0, 0.0]
        return [0.0, 1.0]


class SelectFirstDisambiguator:
    calls = 0

    def choose(
        self,
        surface: str,
        entity_type: EntityType,
        candidates: list[ResolutionCandidate],
    ) -> EntityResolutionJudgment:
        del surface, entity_type
        self.calls += 1
        return EntityResolutionJudgment(
            selected_entity_id=candidates[0].entity_id,
            confidence=0.91,
            rationale_summary="Fixture surfaces denote the same organization",
        )


class FakeEntityJudgeClient:
    def __init__(self, selected_entity_id: str) -> None:
        self.selected_entity_id = selected_entity_id

    def chat_json(self, *_args: object, **_kwargs: object) -> ChatResponse:
        return ChatResponse(
            content=json.dumps(
                {
                    "selected_entity_id": self.selected_entity_id,
                    "confidence": 0.88,
                    "rationale_summary": "The surface is an alias of the candidate",
                }
            ),
            model="fake",
        )


def _chunk(paper_id: str, text: str, page: int = 1, section: str | None = "Method") -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(
            paper_id, page_start=page, page_end=page, text=text, section=section
        ),
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


def test_relation_evidence_is_localized_to_physical_pages() -> None:
    pages = [
        PaperPage(
            paper_id="paper_x",
            page_number=1,
            text="Unrelated preceding material in the same long chunk.",
            char_count=52,
        ),
        PaperPage(
            paper_id="paper_x",
            page_number=2,
            text="Earlier material. The method begins",
            char_count=35,
        ),
        PaperPage(
            paper_id="paper_x",
            page_number=3,
            text="with retrieval and then critiques output.",
            char_count=41,
        ),
    ]
    assert locate_evidence_pages(
        "The method begins with retrieval",
        pages,
        page_start=1,
        page_end=3,
    ) == (2, 3)
    assert locate_evidence_pages(
        "critiques output",
        pages,
        page_start=2,
        page_end=3,
    ) == (3, 3)
    assert (
        locate_evidence_pages(
            "unsupported span",
            pages,
            page_start=2,
            page_end=3,
        )
        is None
    )

    text = "Earlier material. The method begins with retrieval and then critiques output."
    chunk = Chunk(
        chunk_id="chunk_pages",
        paper_id="paper_x",
        text=text,
        page_start=2,
        page_end=3,
        token_count=12,
        content_hash=content_hash(text),
    )
    relation = Relation(
        relation_id="rel_pages",
        subject_surface="method",
        object_surface="retrieval",
        relation_type=RelationType.USES,
        evidence_span="The method begins with retrieval",
        paper_id="paper_x",
        chunk_id=chunk.chunk_id,
        page_number=2,
        page_end=3,
        confidence=0.8,
    )
    localized = localize_relation_to_pages(relation, chunk, pages)
    assert localized is not None
    assert (localized.page_number, localized.page_end) == (2, 3)


def test_entity_alias_fixtures_resolve(repo_root: Path) -> None:
    path = repo_root / "tests" / "fixtures" / "entity_aliases.jsonl"
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
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


def test_embedding_similarity_merges_semantic_candidate() -> None:
    resolver = EntityResolver(embedder=SemanticTestEmbedder())
    first = resolver.register_surface("Alpha Research Laboratory", EntityType.ORGANIZATION)
    second = resolver.register_surface("Alpha semantic passage center", EntityType.ORGANIZATION)
    assert first.entity_id == second.entity_id
    assert resolver.decisions[-1].method == "embedding_similarity"


def test_ambiguous_candidate_can_use_llm_disambiguator() -> None:
    judge = SelectFirstDisambiguator()
    resolver = EntityResolver(
        embedder=SemanticTestEmbedder(),
        disambiguator=judge,
        candidate_floor=0.0,
        ambiguity_margin=1.0,
        embedding_threshold=1.1,
    )
    first = resolver.register_surface("Alpha Research Laboratory", EntityType.ORGANIZATION)
    calls_before_second = judge.calls
    second = resolver.register_surface("Alpha experimental center", EntityType.ORGANIZATION)
    assert first.entity_id == second.entity_id
    assert judge.calls == calls_before_second + 1
    assert resolver.decisions[-1].method == "llm_ambiguous"


def test_llm_disambiguator_returns_structured_bounded_choice() -> None:
    candidate = ResolutionCandidate(
        entity_id="ent_alpha",
        canonical_name="Alpha Lab",
        string_score=0.7,
        embedding_score=0.9,
        combined_score=0.77,
    )
    judge = LLMEntityDisambiguator(
        FakeEntityJudgeClient(candidate.entity_id),  # type: ignore[arg-type]
        max_calls=1,
    )
    result = judge.choose("Alpha Laboratory", EntityType.ORGANIZATION, [candidate])
    assert result.selected_entity_id == candidate.entity_id
    assert result.confidence == 0.88


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
    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    assert meta["corpus_fingerprint"] == ChunkStore.from_processed_dir(processed).fingerprint
    assert meta["graph_schema"] == "graph-v2-physical-page-ranges"

    # A stale/missing build identity must never cause old graph artifacts to be
    # reused against a new canonical chunk store or relation schema.
    result.meta_path.write_text(
        json.dumps({**meta, "corpus_fingerprint": "0" * 32}) + "\n",
        encoding="utf-8",
    )
    current, reason = validate_graph_build_meta(
        result.meta_path,
        corpus_fingerprint=meta["corpus_fingerprint"],
    )
    assert current is False
    assert "fingerprint" in reason
    rebuilt = build_knowledge_graph(processed_dir=processed, force=False)
    rebuilt_meta = json.loads(rebuilt.meta_path.read_text(encoding="utf-8"))
    assert rebuilt_meta["corpus_fingerprint"] == meta["corpus_fingerprint"]
    assert validate_graph_build_meta(
        rebuilt.meta_path,
        corpus_fingerprint=meta["corpus_fingerprint"],
    ) == (True, "ok")

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


def test_extractor_rejects_clause_like_entity_surfaces() -> None:
    chunk = _chunk(
        "paper_noise",
        "We introduce a substantially improved system that uses retrieval for generation.",
    )
    assert extract_from_chunk(chunk) == []


def test_graph_ranking_uses_query_relevance_and_evidence_quality() -> None:
    method = Entity(
        entity_id=make_entity_id("Method", "Self-RAG"),
        entity_type=EntityType.METHOD,
        canonical_name="Self-RAG",
    )
    dataset = Entity(
        entity_id=make_entity_id("Dataset", "HotpotQA"),
        entity_type=EntityType.DATASET,
        canonical_name="HotpotQA",
    )
    retriever_entity = Entity(
        entity_id=make_entity_id("Method", "Generic Retriever"),
        entity_type=EntityType.METHOD,
        canonical_name="Generic Retriever",
    )
    rag_entity = Entity(
        entity_id=make_entity_id("Method", "Retrieval-Augmented Generation"),
        entity_type=EntityType.METHOD,
        canonical_name="Retrieval-Augmented Generation",
        aliases=["RAG"],
    )
    relevant_chunk = _chunk(
        "paper_self_rag",
        "Self-RAG evaluates on HotpotQA using reflection tokens.",
        page=3,
    )
    noise_chunk = _chunk(
        "paper_self_rag",
        "Self-RAG uses a generic retriever for generation.",
        page=8,
    )
    relevant = Relation(
        relation_id="rel_relevant",
        subject_surface="Self-RAG",
        object_surface="HotpotQA",
        subject_entity_id=method.entity_id,
        object_entity_id=dataset.entity_id,
        subject_type=EntityType.METHOD,
        object_type=EntityType.DATASET,
        relation_type=RelationType.EVALUATES_ON,
        evidence_span=relevant_chunk.text,
        paper_id=relevant_chunk.paper_id,
        chunk_id=relevant_chunk.chunk_id,
        page_number=3,
        confidence=0.55,
    )
    noise = Relation(
        relation_id="rel_noise",
        subject_surface="Self-RAG",
        object_surface="Generic Retriever",
        subject_entity_id=method.entity_id,
        object_entity_id=retriever_entity.entity_id,
        subject_type=EntityType.METHOD,
        object_type=EntityType.METHOD,
        relation_type=RelationType.USES,
        evidence_span=noise_chunk.text,
        paper_id=noise_chunk.paper_id,
        chunk_id=noise_chunk.chunk_id,
        page_number=8,
        confidence=0.99,
    )
    graph = KnowledgeGraphStore.from_entities_relations(
        [method, dataset, retriever_entity, rag_entity],
        [relevant, noise],
    )
    papers = [
        Paper(
            paper_id="paper_self_rag",
            title="Self-RAG",
            pdf_path="self-rag.pdf",
            content_hash=content_hash("paper_self_rag"),
        )
    ]
    chunk_store = ChunkStore([relevant_chunk, noise_chunk], papers)
    graph_retriever = GraphRetriever(graph, chunk_store)
    linked = graph_retriever.link_entities("Self-RAG evaluates on HotpotQA")
    assert method.entity_id in linked
    assert dataset.entity_id in linked
    assert rag_entity.entity_id not in linked
    result = graph_retriever.search(
        "How does Self-RAG evaluate on HotpotQA?",
        k=2,
    )
    assert result.hits[0].chunk_id == relevant_chunk.chunk_id
    assert all(hit.chunk_id != noise_chunk.chunk_id for hit in result.hits)
    assert result.debug["score_components"][0]["query_relevance"] > 0

"""Staged entity resolution with embedding candidates and optional LLM arbitration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Protocol

from pydantic import BaseModel, Field, field_validator

from scholar_agent.graph.aliases import ACRONYM_MAP, SEED_ALIASES
from scholar_agent.ids import make_entity_id, normalize_for_id, normalize_text
from scholar_agent.llm.client import ChatMessage, LLMClient
from scholar_agent.llm.structured import request_structured_json_with_retry
from scholar_agent.models.graph import Entity, EntityType
from scholar_agent.retrieval.embeddings import Embedder, HashingEmbedder, cosine_similarity


@dataclass(frozen=True)
class ResolutionCandidate:
    entity_id: str
    canonical_name: str
    string_score: float
    embedding_score: float
    combined_score: float


@dataclass
class ResolutionDecision:
    surface: str
    entity_type: EntityType
    canonical_name: str
    entity_id: str
    method: str
    score: float
    candidates: list[str] = field(default_factory=list)


class EntityResolutionJudgment(BaseModel):
    """Concise LLM decision for an ambiguous canonical-entity match."""

    selected_entity_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_summary: str

    @field_validator("rationale_summary")
    @classmethod
    def _summary_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rationale_summary must be non-empty")
        return value.strip()


class EntityDisambiguator(Protocol):
    def choose(
        self,
        surface: str,
        entity_type: EntityType,
        candidates: list[ResolutionCandidate],
    ) -> EntityResolutionJudgment: ...


@dataclass
class LLMEntityDisambiguator:
    """Use an LLM only when deterministic candidate scores are ambiguous."""

    client: LLMClient
    max_calls: int = 50
    calls: int = 0

    def choose(
        self,
        surface: str,
        entity_type: EntityType,
        candidates: list[ResolutionCandidate],
    ) -> EntityResolutionJudgment:
        if self.calls >= self.max_calls:
            return EntityResolutionJudgment(
                selected_entity_id=None,
                confidence=0.0,
                rationale_summary="LLM resolution budget exhausted",
            )
        self.calls += 1
        allowed_ids = {candidate.entity_id for candidate in candidates}

        def request() -> str:
            payload = [
                {
                    "entity_id": candidate.entity_id,
                    "canonical_name": candidate.canonical_name,
                    "string_score": round(candidate.string_score, 4),
                    "embedding_score": round(candidate.embedding_score, 4),
                }
                for candidate in candidates
            ]
            response = self.client.chat_json(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "Resolve an academic entity surface to one candidate only when they "
                            "refer to the same real entity. Return JSON with selected_entity_id "
                            "(string or null), confidence (0..1), and rationale_summary (brief). "
                            "Do not merge related but distinct models, datasets, or versions."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=json.dumps(
                            {
                                "surface": surface,
                                "entity_type": entity_type.value,
                                "candidates": payload,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ],
                fast=True,
                max_tokens=180,
            )
            return response.content or ""

        judgment = request_structured_json_with_retry(
            request,
            EntityResolutionJudgment,
            max_attempts=2,
        )
        if judgment.selected_entity_id not in allowed_ids:
            return judgment.model_copy(update={"selected_entity_id": None})
        return judgment


@dataclass
class EntityResolver:
    """Resolve normalize → acronym → alias → string+embedding → optional LLM."""

    entities: dict[str, Entity] = field(default_factory=dict)
    alias_to_id: dict[str, str] = field(default_factory=dict)
    decisions: list[ResolutionDecision] = field(default_factory=list)
    embedder: Embedder = field(
        default_factory=lambda: HashingEmbedder(
            dimension=256,
            model_name="entity-hashing-embedder-v1",
        )
    )
    disambiguator: EntityDisambiguator | None = None
    string_threshold: float = 0.92
    embedding_threshold: float = 0.90
    candidate_floor: float = 0.58
    ambiguity_margin: float = 0.06
    max_candidates: int = 8
    _entity_vectors: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for surface, (canonical, entity_type) in SEED_ALIASES.items():
            self.register_surface(surface, entity_type, preferred_canonical=canonical)

    def register_surface(
        self,
        surface: str,
        entity_type: EntityType,
        *,
        preferred_canonical: str | None = None,
    ) -> Entity:
        decision = self.resolve(surface, entity_type, preferred_canonical=preferred_canonical)
        self.decisions.append(decision)
        entity = self.entities[decision.entity_id]
        key = normalize_text(surface)
        self.alias_to_id[key] = entity.entity_id
        if surface not in entity.aliases and normalize_text(surface) != normalize_text(
            entity.canonical_name
        ):
            entity.aliases = list(entity.aliases) + [surface.strip()]
        return entity

    def resolve(
        self,
        surface: str,
        entity_type: EntityType,
        *,
        preferred_canonical: str | None = None,
    ) -> ResolutionDecision:
        raw = surface.strip()
        if not raw:
            raise ValueError("empty entity surface")
        norm = normalize_text(raw)
        expanded_norm = normalize_text(ACRONYM_MAP.get(norm, norm))

        for key in (norm, expanded_norm):
            if key in self.alias_to_id:
                entity = self.entities[self.alias_to_id[key]]
                return self._decision(raw, entity, "exact_alias", 1.0, [entity.entity_id])
            if key in SEED_ALIASES:
                canonical, known_type = SEED_ALIASES[key]
                return self._create_or_get(
                    canonical,
                    known_type,
                    raw,
                    method="seed_alias",
                    score=1.0,
                )

        candidates = self._rank_candidates(raw, entity_type)
        if candidates:
            best = candidates[0]
            best_entity = self.entities[best.entity_id]

            if best.string_score >= self.string_threshold:
                return self._merge_decision(raw, best_entity, "string_similarity", best.string_score, candidates)

            ambiguous = (
                best.combined_score >= self.candidate_floor
                and (
                    len(candidates) == 1
                    or best.combined_score - candidates[1].combined_score <= self.ambiguity_margin
                )
            )
            if ambiguous and self.disambiguator is not None:
                judgment = self.disambiguator.choose(raw, entity_type, candidates[:5])
                if judgment.selected_entity_id:
                    selected = self.entities[judgment.selected_entity_id]
                    return self._merge_decision(
                        raw,
                        selected,
                        "llm_ambiguous",
                        judgment.confidence,
                        candidates,
                    )

            if best.embedding_score >= self.embedding_threshold and best.string_score >= 0.35:
                return self._merge_decision(
                    raw,
                    best_entity,
                    "embedding_similarity",
                    best.embedding_score,
                    candidates,
                )

        canonical = preferred_canonical or raw
        return self._create_or_get(
            canonical,
            entity_type,
            raw,
            method="new_entity",
            score=1.0,
        )

    def _rank_candidates(
        self,
        surface: str,
        entity_type: EntityType,
    ) -> list[ResolutionCandidate]:
        norm = normalize_text(surface)
        string_ranked: list[tuple[float, Entity]] = []
        for entity in self.entities.values():
            if entity.entity_type != entity_type:
                continue
            score = max(
                SequenceMatcher(None, norm, normalize_text(entity.canonical_name)).ratio(),
                max(
                    (
                        SequenceMatcher(None, norm, normalize_text(alias)).ratio()
                        for alias in entity.aliases
                    ),
                    default=0.0,
                ),
            )
            string_ranked.append((score, entity))
        string_ranked.sort(key=lambda item: (-item[0], item[1].entity_id))

        query_vector = self.embedder.embed_query(surface)
        candidates: list[ResolutionCandidate] = []
        for string_score, entity in string_ranked[: self.max_candidates]:
            entity_vector = self._entity_vectors.get(entity.entity_id)
            if entity_vector is None:
                entity_vector = self.embedder.embed_query(entity.canonical_name)
                self._entity_vectors[entity.entity_id] = entity_vector
            embedding_score = cosine_similarity(query_vector, entity_vector)
            combined = 0.65 * string_score + 0.35 * max(0.0, embedding_score)
            candidates.append(
                ResolutionCandidate(
                    entity_id=entity.entity_id,
                    canonical_name=entity.canonical_name,
                    string_score=string_score,
                    embedding_score=embedding_score,
                    combined_score=combined,
                )
            )
        candidates.sort(key=lambda item: (-item.combined_score, item.entity_id))
        return candidates

    def _merge_decision(
        self,
        surface: str,
        entity: Entity,
        method: str,
        score: float,
        candidates: list[ResolutionCandidate],
    ) -> ResolutionDecision:
        self.alias_to_id[normalize_text(surface)] = entity.entity_id
        return self._decision(
            surface,
            entity,
            method,
            score,
            [candidate.entity_id for candidate in candidates[:5]],
        )

    @staticmethod
    def _decision(
        surface: str,
        entity: Entity,
        method: str,
        score: float,
        candidates: list[str],
    ) -> ResolutionDecision:
        return ResolutionDecision(
            surface=surface,
            entity_type=entity.entity_type,
            canonical_name=entity.canonical_name,
            entity_id=entity.entity_id,
            method=method,
            score=score,
            candidates=candidates,
        )

    def _create_or_get(
        self,
        canonical: str,
        entity_type: EntityType,
        surface: str,
        *,
        method: str,
        score: float,
    ) -> ResolutionDecision:
        entity_id = make_entity_id(entity_type.value, canonical)
        if entity_id not in self.entities:
            self.entities[entity_id] = Entity(
                entity_id=entity_id,
                entity_type=entity_type,
                canonical_name=canonical,
                aliases=[],
            )
            self._entity_vectors[entity_id] = self.embedder.embed_query(canonical)
        self.alias_to_id[normalize_text(surface)] = entity_id
        self.alias_to_id[normalize_text(canonical)] = entity_id
        self.alias_to_id[normalize_for_id(canonical).replace("_", " ")] = entity_id
        return self._decision(
            surface,
            self.entities[entity_id],
            method,
            score,
            [entity_id],
        )

    def all_entities(self) -> list[Entity]:
        return list(self.entities.values())

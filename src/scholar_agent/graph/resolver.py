"""Staged entity resolution (no LLM required for the default path)."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from scholar_agent.graph.aliases import ACRONYM_MAP, SEED_ALIASES
from scholar_agent.ids import make_entity_id, normalize_for_id, normalize_text
from scholar_agent.models.graph import Entity, EntityType


@dataclass
class ResolutionDecision:
    surface: str
    entity_type: EntityType
    canonical_name: str
    entity_id: str
    method: str
    score: float
    candidates: list[str] = field(default_factory=list)


@dataclass
class EntityResolver:
    """Staged resolver: normalize → acronym → alias → string similarity."""

    entities: dict[str, Entity] = field(default_factory=dict)
    alias_to_id: dict[str, str] = field(default_factory=dict)
    decisions: list[ResolutionDecision] = field(default_factory=list)
    string_threshold: float = 0.92

    def __post_init__(self) -> None:
        # Load seed aliases as provisional entities
        for surface, (canonical, etype) in SEED_ALIASES.items():
            self.register_surface(surface, etype, preferred_canonical=canonical)

    def register_surface(
        self,
        surface: str,
        entity_type: EntityType,
        *,
        preferred_canonical: str | None = None,
    ) -> Entity:
        """Resolve or create a canonical entity for a surface form."""
        decision = self.resolve(surface, entity_type, preferred_canonical=preferred_canonical)
        self.decisions.append(decision)
        entity = self.entities[decision.entity_id]
        # Track alias
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

        # Stage 1: normalize
        norm = normalize_text(raw)

        # Stage 2: acronym expansion
        expanded = ACRONYM_MAP.get(norm, norm)
        expanded_norm = normalize_text(expanded)

        # Stage 3: exact alias match
        for key in (norm, expanded_norm):
            if key in self.alias_to_id:
                eid = self.alias_to_id[key]
                ent = self.entities[eid]
                return ResolutionDecision(
                    surface=raw,
                    entity_type=ent.entity_type,
                    canonical_name=ent.canonical_name,
                    entity_id=ent.entity_id,
                    method="exact_alias",
                    score=1.0,
                    candidates=[ent.entity_id],
                )
            if key in SEED_ALIASES:
                canonical, etype = SEED_ALIASES[key]
                return self._create_or_get(canonical, etype, raw, method="seed_alias", score=1.0)

        # Stage 4: string similarity against existing same-type entities
        candidates: list[tuple[float, Entity]] = []
        for ent in self.entities.values():
            if ent.entity_type != entity_type:
                continue
            score = max(
                SequenceMatcher(None, norm, normalize_text(ent.canonical_name)).ratio(),
                max(
                    (
                        SequenceMatcher(None, norm, normalize_text(a)).ratio()
                        for a in ent.aliases
                    ),
                    default=0.0,
                ),
            )
            if score >= self.string_threshold:
                candidates.append((score, ent))
        candidates.sort(key=lambda x: -x[0])
        if candidates:
            best_score, best = candidates[0]
            self.alias_to_id[norm] = best.entity_id
            return ResolutionDecision(
                surface=raw,
                entity_type=best.entity_type,
                canonical_name=best.canonical_name,
                entity_id=best.entity_id,
                method="string_similarity",
                score=best_score,
                candidates=[e.entity_id for _, e in candidates[:5]],
            )

        # Stage 5/6: LLM for ambiguous pairs is optional and skipped offline
        # Stage 7: create new canonical entity
        canonical = preferred_canonical or raw.strip()
        # Prefer title-case-ish canonical for multi-word methods
        if preferred_canonical is None and raw.isupper() and len(raw) <= 8:
            canonical = raw  # keep acronyms as-is
        return self._create_or_get(
            canonical, entity_type, raw, method="new_entity", score=1.0
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
        eid = make_entity_id(entity_type.value, canonical)
        if eid not in self.entities:
            self.entities[eid] = Entity(
                entity_id=eid,
                entity_type=entity_type,
                canonical_name=canonical,
                aliases=[],
            )
        self.alias_to_id[normalize_text(surface)] = eid
        self.alias_to_id[normalize_text(canonical)] = eid
        # also register slug form
        self.alias_to_id[normalize_for_id(canonical).replace("_", " ")] = eid
        ent = self.entities[eid]
        return ResolutionDecision(
            surface=surface,
            entity_type=ent.entity_type,
            canonical_name=ent.canonical_name,
            entity_id=ent.entity_id,
            method=method,
            score=score,
            candidates=[eid],
        )

    def all_entities(self) -> list[Entity]:
        return list(self.entities.values())

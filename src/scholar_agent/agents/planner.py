"""Structured query planning with a deterministic, offline-safe fallback."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from scholar_agent.graph.aliases import SEED_ALIASES
from scholar_agent.ids import (
    make_entity_id,
    make_sub_question_id,
    normalize_for_id,
    normalize_text,
)
from scholar_agent.llm.client import ChatMessage, LLMClient
from scholar_agent.llm.structured import StructuredOutputError, parse_structured_json
from scholar_agent.logging import get_logger
from scholar_agent.models.base import QueryType, TokenUsage
from scholar_agent.models.graph import EntityType
from scholar_agent.models.planning import (
    AnswerRequirement,
    PlanDraft,
    PlannedEntity,
    QueryPlan,
    SubQuestion,
    SubQuestionStatus,
)
from scholar_agent.retrieval.router import classify_query_type

logger = get_logger(__name__)

_ANCHOR_TOKEN = re.compile(r"\b[A-Za-z0-9]+(?:[-_=][A-Za-z0-9]+)*\b")
_QUESTION_WORDS = {
    "According",
    "Compare",
    "How",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
}
_COMPARISON_PATTERNS = (
    re.compile(
        r"^\s*(?:(?:compare|comparison\s+of)\s+)?"
        r"(?P<left>.+?)\s+(?:versus|vs\.?)\s+(?P<right>.+)$",
        re.I | re.S,
    ),
    re.compile(
        r"^\s*compare\s+(?P<left>.+?)\s+and\s+(?P<right>.+)$",
        re.I | re.S,
    ),
    re.compile(
        r"(?:key\s+)?differences?\s+between\s+"
        r"(?P<left>.+?)\s+and\s+(?P<right>.+)$",
        re.I | re.S,
    ),
)
_ENTITY_TRAILING_BOUNDARY = re.compile(
    r"(?:"
    r"[;?!\n]|"
    r"\.(?=\s|$)|"
    r",\s*(?=(?:explain|describe|discuss|analy[sz]e|identify|summari[sz]e|"
    r"outline|evaluate|compare)\b)|"
    r"\s+(?:and\s+)?(?=(?:explain|describe|discuss|analy[sz]e|identify|"
    r"summari[sz]e|outline|evaluate)\b)|"
    r"\s+(?:in\s+terms\s+of|with\s+respect\s+to|regarding)\s+"
    r")",
    re.I,
)
_DIMENSION_DESCRIPTIONS = {
    "retrieval_trigger": "How and when retrieval is triggered",
    "correction_mechanism": "How retrieved evidence is evaluated and corrected",
    "key_differences": "The key differences between the target entities",
    "definition_or_fact": "The requested definition or factual answer",
    "relation": "The requested relationship and its supporting evidence",
    "main_themes": "The main themes across the requested literature",
    "central_methods": "The central methods or systems",
    "open_challenges": "The limitations and open challenges",
}
_LLM_PROMPT_VERSION = "planner-plan-draft-v1"
_DETERMINISTIC_PROMPT_VERSION = "planner-deterministic-v2"


class PlannerLLMError(RuntimeError):
    """A sanitized strict-mode LLM planning failure."""


def extract_answer_anchors(query: str) -> list[str]:
    """Extract exact named/version anchors that evidence must preserve.

    The rule is syntax based rather than corpus- or benchmark-specific: years,
    acronyms, camel-case names, and mixed-case/versioned hyphenated names are
    retained. This prevents generic lexical overlap from satisfying an entity-
    specific question.
    """
    anchors: list[str] = []
    for token in _ANCHOR_TOKEN.findall(query):
        if token in _QUESTION_WORDS:
            continue
        letters = "".join(char for char in token if char.isalpha())
        is_year = bool(re.fullmatch(r"(?:19|20)\d{2}", token))
        is_acronym = len(letters) >= 2 and letters.isupper()
        is_camel = any(char.isupper() for char in token[1:]) and any(
            char.islower() for char in token
        )
        is_versioned_name = ("-" in token or "=" in token or "_" in token) and (
            any(char.isupper() for char in token) or any(char.isdigit() for char in token)
        )
        if (is_year or is_acronym or is_camel or is_versioned_name) and token not in anchors:
            anchors.append(token)
    return anchors


@dataclass
class Planner:
    """Produce a validated :class:`QueryPlan`.

    With no ``llm`` this class is fully deterministic and performs no provider
    calls. When an LLM is configured, only a constrained ``PlanDraft`` comes
    from the model; stable IDs and all cross-reference validation remain local.
    """

    llm: LLMClient | None = None
    strict_llm: bool = False
    last_backend: str = field(default="deterministic", init=False)
    last_model: str | None = field(default=None, init=False)
    last_fallback_reason: str | None = field(default=None, init=False)
    last_token_usage: TokenUsage = field(default_factory=TokenUsage, init=False)
    last_prompt_version: str = field(default=_DETERMINISTIC_PROMPT_VERSION, init=False)

    def plan(self, query: str) -> QueryPlan:
        query = query.strip()
        if not query:
            raise ValueError("query must be non-empty")

        self._reset_run_metadata()
        if self.llm is not None:
            try:
                return self._plan_with_llm(query)
            except Exception as exc:
                reason = _safe_failure_reason(exc)
                self.last_fallback_reason = reason
                if self.strict_llm:
                    raise PlannerLLMError(f"LLM planner failed: {reason}") from exc
                self.last_backend = "deterministic"
                logger.warning("LLM planner degraded to deterministic reason=%s", reason)

        return self._plan_deterministic(query)

    def _reset_run_metadata(self) -> None:
        self.last_backend = "deterministic"
        self.last_model = None
        self.last_fallback_reason = None
        self.last_token_usage = TokenUsage()
        self.last_prompt_version = _DETERMINISTIC_PROMPT_VERSION

    def _plan_with_llm(self, query: str) -> QueryPlan:
        if self.llm is None:  # pragma: no cover - guarded by ``plan``
            raise PlannerLLMError("LLM planner was invoked without a client")

        self.last_backend = "llm"
        self.last_prompt_version = _LLM_PROMPT_VERSION
        response = self.llm.chat_json(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "Create a compact academic research plan as one JSON object. "
                        "Return exactly these fields: answer_type (string), "
                        "target_entities (array of user-visible entity names), "
                        "answer_requirements (array of concise dimensions), "
                        "sub_questions (array of objects with question, query_type, "
                        "target_entities, requirements, dimension, required_evidence), "
                        "and expected_source_diversity (positive integer). "
                        "query_type must be semantic, keyword, comparison, relational, "
                        "or synthesis. For comparisons, include both entities and ensure "
                        "every requirement is covered for both entities; use separate "
                        "entity-specific questions where needed. Only include target "
                        "entities explicitly named in the user's query. Do not invent "
                        "evidence, citations, IDs, entities, or facts."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=json.dumps({"query": query}, ensure_ascii=False),
                ),
            ],
            fast=True,
            max_tokens=1200,
        )
        self.last_model = response.model
        self.last_token_usage = response.usage.model_copy(deep=True)
        draft = parse_structured_json(response.content or "", PlanDraft)
        plan = self._finalize_draft(query, draft)
        logger.info(
            "planned backend=llm model=%s sub_questions=%s",
            response.model,
            len(plan.sub_questions),
        )
        return plan

    def _finalize_draft(self, query: str, draft: PlanDraft) -> QueryPlan:
        answer_type = normalize_for_id(draft.answer_type)
        entities = _unique_entities(
            [_resolve_planned_entity(surface) for surface in draft.target_entities]
        )
        entities = _anchor_draft_entities(query, answer_type, entities)
        if answer_type == "comparison" and len(entities) != 2:
            raise ValueError("comparison PlanDraft must contain exactly two target entities")
        if not draft.sub_questions:
            raise ValueError("PlanDraft must contain at least one sub-question")
        if not draft.answer_requirements:
            raise ValueError("PlanDraft must contain at least one answer requirement")

        all_entity_ids = [entity.id for entity in entities]
        requirements: list[AnswerRequirement] = []
        for raw_dimension in draft.answer_requirements:
            key = _canonical_dimension(raw_dimension)
            if key in {requirement.key for requirement in requirements}:
                continue
            requirements.append(
                _answer_requirement(
                    key,
                    all_entity_ids if entities else [],
                    description=raw_dimension,
                )
            )

        entity_lookup = _entity_lookup(entities)
        requirement_lookup: dict[str, str] = {}
        for requirement in requirements:
            requirement_lookup[normalize_text(requirement.key)] = requirement.key
            requirement_lookup[normalize_for_id(requirement.description)] = requirement.key
        for raw_dimension in draft.answer_requirements:
            requirement_lookup[normalize_for_id(raw_dimension)] = _canonical_dimension(raw_dimension)

        plan_seed = normalize_text(query)[:64]
        sub_questions: list[SubQuestion] = []
        for index, draft_question in enumerate(draft.sub_questions):
            target_ids = _resolve_draft_references(
                draft_question.target_entities,
                entity_lookup,
                reference_kind="entity",
            )
            requirement_keys = _resolve_draft_references(
                draft_question.requirements,
                requirement_lookup,
                reference_kind="requirement",
                normalizer=normalize_for_id,
            )
            dimension = (
                _canonical_dimension(draft_question.dimension)
                if draft_question.dimension
                else None
            )
            if dimension is not None and dimension not in {item.key for item in requirements}:
                raise ValueError(f"sub-question references unknown dimension: {dimension}")
            if dimension is not None and dimension not in requirement_keys:
                requirement_keys.append(dimension)
            required_evidence = list(draft_question.required_evidence)
            if not required_evidence:
                required_evidence = [
                    *(entity.canonical_name for entity in entities if entity.id in target_ids),
                    *(requirement_keys or ([dimension] if dimension else [])),
                    "supporting passage",
                ]
            sub_questions.append(
                SubQuestion(
                    id=make_sub_question_id(plan_seed, draft_question.question, index),
                    question=draft_question.question,
                    query_type=draft_question.query_type,
                    required_evidence=required_evidence,
                    status=SubQuestionStatus.PENDING,
                    target_entity_ids=target_ids,
                    requirement_keys=requirement_keys,
                    dimension=dimension,
                )
            )

        _validate_draft_coverage(entities, requirements, sub_questions)
        return QueryPlan(
            original_query=query,
            answer_type=answer_type,
            sub_questions=self._with_answer_anchors(query, sub_questions),
            expected_source_diversity=draft.expected_source_diversity,
            target_entities=entities,
            answer_requirements=requirements,
        )

    def _plan_deterministic(self, query: str) -> QueryPlan:
        qtype, signals = classify_query_type(query)
        if len(_parse_comparison_entities(query)) == 2:
            qtype = QueryType.COMPARISON
            if "comparison_parser" not in signals:
                signals.append("comparison_parser")
        plan_seed = normalize_text(query)[:64]

        if qtype == QueryType.COMPARISON:
            plan = self._plan_comparison(query, plan_seed)
        elif qtype == QueryType.SYNTHESIS:
            plan = self._plan_synthesis(query, plan_seed)
        elif qtype == QueryType.RELATIONAL:
            entities = _extract_known_entities(query)
            entity_ids = [entity.id for entity in entities]
            requirements = [_answer_requirement("relation", entity_ids)]
            sub_questions = [
                SubQuestion(
                    id=make_sub_question_id(plan_seed, query, 0),
                    question=query,
                    query_type=QueryType.RELATIONAL,
                    required_evidence=["relation", "entities", "supporting passage"],
                    status=SubQuestionStatus.PENDING,
                    target_entity_ids=entity_ids,
                    requirement_keys=["relation"],
                    dimension="relation",
                )
            ]
            plan = QueryPlan(
                original_query=query,
                answer_type="relational",
                sub_questions=self._with_answer_anchors(query, sub_questions),
                expected_source_diversity=2,
                target_entities=entities,
                answer_requirements=requirements,
            )
        else:
            entities = _extract_known_entities(query)
            entity_ids = [entity.id for entity in entities]
            requirements = [_answer_requirement("definition_or_fact", entity_ids)]
            sub_questions = [
                SubQuestion(
                    id=make_sub_question_id(plan_seed, query, 0),
                    question=query,
                    query_type=qtype,
                    required_evidence=["definition_or_fact", "supporting passage"],
                    status=SubQuestionStatus.PENDING,
                    target_entity_ids=entity_ids,
                    requirement_keys=["definition_or_fact"],
                    dimension="definition_or_fact",
                )
            ]
            answer_type = "factual" if qtype == QueryType.KEYWORD else "semantic"
            plan = QueryPlan(
                original_query=query,
                answer_type=answer_type,
                sub_questions=self._with_answer_anchors(query, sub_questions),
                expected_source_diversity=1,
                target_entities=entities,
                answer_requirements=requirements,
            )

        logger.info(
            "planned backend=deterministic query_type=%s sub_questions=%s signals=%s",
            qtype.value,
            len(plan.sub_questions),
            signals[:5],
        )
        return plan

    def _plan_comparison(self, query: str, plan_seed: str) -> QueryPlan:
        surfaces = _parse_comparison_entities(query)
        if len(surfaces) != 2:
            entities = _extract_known_entities(query)
            entity_ids = [entity.id for entity in entities]
            requirements = [_answer_requirement("key_differences", entity_ids)]
            questions = [
                SubQuestion(
                    id=make_sub_question_id(plan_seed, query, 0),
                    question=query,
                    query_type=QueryType.COMPARISON,
                    required_evidence=["comparison", "both sides"],
                    status=SubQuestionStatus.PENDING,
                    target_entity_ids=entity_ids,
                    requirement_keys=["key_differences"],
                    dimension="key_differences",
                ),
                SubQuestion(
                    id=make_sub_question_id(
                        plan_seed,
                        f"Key differences and trade-offs for: {query}",
                        1,
                    ),
                    question=f"Key differences and trade-offs for: {query}",
                    query_type=QueryType.COMPARISON,
                    required_evidence=["trade-offs"],
                    status=SubQuestionStatus.PENDING,
                    target_entity_ids=entity_ids,
                    requirement_keys=["key_differences"],
                    dimension="key_differences",
                ),
            ]
            return QueryPlan(
                original_query=query,
                answer_type="comparison",
                sub_questions=self._with_answer_anchors(query, questions),
                expected_source_diversity=2,
                target_entities=entities,
                answer_requirements=requirements,
            )

        entities = _unique_entities(
            [_resolve_planned_entity(surface) for surface in surfaces]
        )
        if len(entities) != 2:
            raise ValueError("comparison entities must resolve to two distinct identities")
        entity_ids = [entity.id for entity in entities]
        dimensions = _comparison_dimensions(query, entities)
        requirements = [
            _answer_requirement(dimension, entity_ids) for dimension in dimensions
        ]

        sub_questions: list[SubQuestion] = []
        # Start with the cross-entity question so even a very small global
        # budget exercises the comparison retrieval policy. Entity-specific
        # dimensions follow and carry their own explicit assignments.
        question_dimensions = list(dimensions)
        if "key_differences" in question_dimensions:
            question_dimensions.remove("key_differences")
            question_dimensions.insert(0, "key_differences")
        for dimension in question_dimensions:
            if dimension == "key_differences":
                left, right = (_display_name(entity) for entity in entities)
                question = f"What are the key differences between {left} and {right}?"
                sub_questions.append(
                    SubQuestion(
                        id=make_sub_question_id(plan_seed, question, len(sub_questions)),
                        question=question,
                        query_type=QueryType.COMPARISON,
                        required_evidence=["comparison", "both sides"],
                        status=SubQuestionStatus.PENDING,
                        target_entity_ids=entity_ids,
                        requirement_keys=[dimension],
                        dimension=dimension,
                    )
                )
                continue

            for entity in entities:
                label = _display_name(entity)
                aspect = (
                    "retrieval trigger"
                    if dimension == "retrieval_trigger"
                    else "correction mechanism"
                )
                question = f"What is the {aspect} used by {label}?"
                sub_questions.append(
                    SubQuestion(
                        id=make_sub_question_id(plan_seed, question, len(sub_questions)),
                        question=question,
                        query_type=QueryType.SEMANTIC,
                        required_evidence=[
                            "definition",
                            entity.canonical_name,
                            "supporting passage",
                        ],
                        status=SubQuestionStatus.PENDING,
                        target_entity_ids=[entity.id],
                        requirement_keys=[dimension],
                        dimension=dimension,
                    )
                )

        return QueryPlan(
            original_query=query,
            answer_type="comparison",
            sub_questions=self._with_answer_anchors(query, sub_questions),
            expected_source_diversity=2,
            target_entities=entities,
            answer_requirements=requirements,
        )

    def _plan_synthesis(self, query: str, plan_seed: str) -> QueryPlan:
        entities = _extract_known_entities(query)
        entity_ids = [entity.id for entity in entities]
        specifications = [
            (query, QueryType.SYNTHESIS, "main_themes", ["main themes"]),
            (
                f"What methods or systems are most central to: {query}",
                QueryType.KEYWORD,
                "central_methods",
                ["methods", "systems"],
            ),
            (
                f"What open challenges remain regarding: {query}",
                QueryType.SEMANTIC,
                "open_challenges",
                ["limitations", "open problems"],
            ),
        ]
        sub_questions = [
            SubQuestion(
                id=make_sub_question_id(plan_seed, question, index),
                question=question,
                query_type=query_type,
                required_evidence=required_evidence,
                status=SubQuestionStatus.PENDING,
                target_entity_ids=entity_ids,
                requirement_keys=[dimension],
                dimension=dimension,
            )
            for index, (question, query_type, dimension, required_evidence) in enumerate(
                specifications
            )
        ]
        return QueryPlan(
            original_query=query,
            answer_type="synthesis",
            sub_questions=self._with_answer_anchors(query, sub_questions),
            expected_source_diversity=3,
            target_entities=entities,
            answer_requirements=[
                _answer_requirement(dimension, entity_ids)
                for dimension in ("main_themes", "central_methods", "open_challenges")
            ],
        )

    def _with_answer_anchors(
        self,
        query: str,
        sub_questions: list[SubQuestion],
    ) -> list[SubQuestion]:
        anchors = extract_answer_anchors(query)
        if not anchors:
            return sub_questions
        anchored: list[SubQuestion] = []
        for sub_question in sub_questions:
            required_evidence = list(sub_question.required_evidence)
            # A cross-entity comparison can be supported by separate passages;
            # requiring every exact entity token in one shared passage would
            # defeat the structured target assignments.
            question_anchors = (
                []
                if len(sub_question.target_entity_ids) > 1
                else extract_answer_anchors(sub_question.question) or anchors
            )
            for anchor in question_anchors:
                requirement = f"anchor:{anchor}"
                if requirement not in required_evidence:
                    required_evidence.append(requirement)
            anchored.append(
                sub_question.model_copy(update={"required_evidence": required_evidence})
            )
        return anchored


def _parse_comparison_entities(query: str) -> list[str]:
    for pattern in _COMPARISON_PATTERNS:
        match = pattern.search(query)
        if match is None:
            continue
        left = _clean_entity_surface(match.group("left"))
        right = _clean_entity_surface(match.group("right"))
        if left and right:
            return [left, right]
    return []


def _clean_entity_surface(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(
        r"^(?:compare|comparison\s+of|(?:key\s+)?differences?\s+between)\s+",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = _ENTITY_TRAILING_BOUNDARY.split(cleaned, maxsplit=1)[0]
    return cleaned.strip(" \t\r\n`'\".,:;?!")


def _resolve_planned_entity(surface: str) -> PlannedEntity:
    cleaned = _clean_entity_surface(surface)
    normalized = normalize_text(cleaned)

    # "CRAG -- Comprehensive RAG Benchmark" is a benchmark name, not the
    # exact ``crag`` method alias. This guard runs before parenthetical/alias
    # expansion so generic RAG tokens cannot collapse the two identities.
    if "comprehensive rag benchmark" in normalized:
        canonical_name = "Comprehensive RAG Benchmark"
        entity_type = EntityType.DATASET
    else:
        candidates = [normalized]
        parenthetical = re.fullmatch(r"(.+?)\s*\(([^()]*)\)", cleaned)
        if parenthetical:
            candidates.extend(
                [
                    normalize_text(parenthetical.group(1)),
                    normalize_text(parenthetical.group(2)),
                ]
            )
        known = next((SEED_ALIASES[item] for item in candidates if item in SEED_ALIASES), None)
        if known is None:
            canonical_name = cleaned
            entity_type = EntityType.METHOD
        else:
            canonical_name, entity_type = known

    aliases = [cleaned, canonical_name]
    aliases.extend(
        alias
        for alias, (known_canonical, _known_type) in SEED_ALIASES.items()
        if normalize_text(known_canonical) == normalize_text(canonical_name)
    )
    return PlannedEntity(
        id=make_entity_id(entity_type.value, canonical_name),
        surface_name=cleaned,
        canonical_name=canonical_name,
        aliases=aliases,
    )


def _extract_known_entities(query: str) -> list[PlannedEntity]:
    normalized_query = normalize_text(query)
    matches: list[tuple[int, int, str]] = []
    for alias in sorted(SEED_ALIASES, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        for match in pattern.finditer(normalized_query):
            if any(match.start() < end and match.end() > start for start, end, _ in matches):
                continue
            matches.append((match.start(), match.end(), alias))
    matches.sort(key=lambda item: item[0])
    return _unique_entities([_resolve_planned_entity(alias) for _, _, alias in matches])


def _unique_entities(entities: list[PlannedEntity]) -> list[PlannedEntity]:
    unique: list[PlannedEntity] = []
    by_id: dict[str, int] = {}
    for entity in entities:
        if entity.id not in by_id:
            by_id[entity.id] = len(unique)
            unique.append(entity)
            continue
        index = by_id[entity.id]
        existing = unique[index]
        unique[index] = PlannedEntity(
            id=existing.id,
            surface_name=existing.surface_name,
            canonical_name=existing.canonical_name,
            aliases=[*existing.aliases, *entity.aliases],
        )
    return unique


def _anchor_draft_entities(
    query: str,
    answer_type: str,
    entities: list[PlannedEntity],
) -> list[PlannedEntity]:
    """Reject entity substitutions before an LLM draft becomes a query plan.

    Comparison syntax is authoritative: aliases are resolved locally and the
    LLM must name exactly those two canonical identities. For other questions,
    every emitted entity must have a complete surface/canonical/alias match in
    the original query. Known aliases use the longest-match local extractor so
    generic ``RAG`` cannot claim a more specific ``Self-RAG``/``GraphRAG`` hit.
    """
    comparison_surfaces = _parse_comparison_entities(query)
    if len(comparison_surfaces) == 2:
        if answer_type != "comparison":
            raise ValueError("PlanDraft answer type disagrees with comparison syntax")
        expected = _unique_entities(
            [_resolve_planned_entity(surface) for surface in comparison_surfaces]
        )
        if len(expected) != 2 or {item.id for item in entities} != {
            item.id for item in expected
        }:
            raise ValueError("PlanDraft comparison entities do not match the query")
        # Preserve query spelling and order rather than LLM-authored surfaces.
        return expected

    if answer_type == "comparison":
        raise ValueError("PlanDraft comparison entities cannot be anchored to the query")

    known_query_ids = {entity.id for entity in _extract_known_entities(query)}
    known_entity_ids = {
        make_entity_id(entity_type.value, canonical_name)
        for canonical_name, entity_type in SEED_ALIASES.values()
    }
    for entity in entities:
        if entity.id in known_entity_ids:
            anchored = entity.id in known_query_ids
        else:
            anchored = any(
                _alias_boundary_match(query, candidate)
                for candidate in [
                    entity.surface_name,
                    entity.canonical_name,
                    *entity.aliases,
                ]
            )
        if not anchored:
            raise ValueError("PlanDraft entity is not anchored in the query")
    return entities


def _alias_boundary_match(query: str, alias: str) -> bool:
    normalized_query = normalize_text(query)
    normalized_alias = normalize_text(alias)
    if not normalized_alias:
        return False
    return bool(
        re.search(
            rf"(?<!\w){re.escape(normalized_alias)}(?!\w)",
            normalized_query,
        )
    )


def _display_name(entity: PlannedEntity) -> str:
    # Preserve the user's spelling in retrieval queries. In particular, writing
    # the canonical ``Corrective RAG`` into a sub-question would accidentally
    # activate the router's corrective-pass cue; canonical identity remains
    # available on ``PlannedEntity.canonical_name``.
    return entity.surface_name


def _comparison_dimensions(
    query: str,
    entities: list[PlannedEntity],
) -> list[str]:
    normalized = normalize_text(query)
    dimensions: list[str] = []
    if re.search(r"\bretriev(?:al|e|ing)\b|\btrigger", normalized):
        dimensions.append("retrieval_trigger")
    if re.search(r"\bcorrect(?:ion|ive|ing)?\b|\brefin(?:e|ement)\b", normalized):
        dimensions.append("correction_mechanism")

    canonical_names = {normalize_text(entity.canonical_name) for entity in entities}
    self_crag_comparison = canonical_names == {
        normalize_text("Self-RAG"),
        normalize_text("Corrective RAG"),
    }
    if self_crag_comparison:
        if "retrieval_trigger" not in dimensions:
            dimensions.append("retrieval_trigger")
        if "correction_mechanism" not in dimensions:
            dimensions.append("correction_mechanism")
    dimensions.append("key_differences")
    return dimensions


def _canonical_dimension(raw: str) -> str:
    normalized = normalize_for_id(raw)
    if ("retriev" in normalized and "trigger" in normalized) or normalized == "retrieval":
        return "retrieval_trigger"
    if any(term in normalized for term in ("correct", "refine")):
        return "correction_mechanism"
    if any(term in normalized for term in ("difference", "comparison", "trade_off")):
        return "key_differences"
    return normalized


def _answer_requirement(
    key: str,
    target_entity_ids: list[str],
    *,
    description: str | None = None,
) -> AnswerRequirement:
    return AnswerRequirement(
        key=key,
        description=_DIMENSION_DESCRIPTIONS.get(key, description or key.replace("_", " ")),
        target_entity_ids=target_entity_ids,
    )


def _entity_lookup(entities: list[PlannedEntity]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for entity in entities:
        for name in [entity.surface_name, entity.canonical_name, *entity.aliases]:
            lookup[normalize_text(name)] = entity.id
    return lookup


def _resolve_draft_references(
    references: list[str],
    lookup: dict[str, str],
    *,
    reference_kind: str,
    normalizer: Callable[[str], str] = normalize_text,
) -> list[str]:
    resolved: list[str] = []
    for reference in references:
        key = normalizer(reference)
        if key not in lookup:
            raise ValueError(f"PlanDraft references unknown {reference_kind}")
        value = lookup[key]
        if value not in resolved:
            resolved.append(value)
    return resolved


def _validate_draft_coverage(
    entities: list[PlannedEntity],
    requirements: list[AnswerRequirement],
    sub_questions: list[SubQuestion],
) -> None:
    referenced_entities = {
        entity_id for sub_question in sub_questions for entity_id in sub_question.target_entity_ids
    }
    referenced_requirements = {
        key for sub_question in sub_questions for key in sub_question.requirement_keys
    }
    if referenced_entities != {entity.id for entity in entities}:
        raise ValueError("PlanDraft sub-questions do not cover every target entity")
    if referenced_requirements != {requirement.key for requirement in requirements}:
        raise ValueError("PlanDraft sub-questions do not cover every answer requirement")

    covered_pairs = {
        (requirement_key, entity_id)
        for sub_question in sub_questions
        for requirement_key in sub_question.requirement_keys
        for entity_id in sub_question.target_entity_ids
    }
    for requirement in requirements:
        for entity_id in requirement.target_entity_ids:
            if (requirement.key, entity_id) not in covered_pairs:
                raise ValueError(
                    "PlanDraft does not cover every requirement/entity combination"
                )


def _safe_failure_reason(exc: Exception) -> str:
    if isinstance(exc, StructuredOutputError):
        return "invalid_structured_output"
    if isinstance(exc, ValidationError):
        return "invalid_plan_draft"
    if isinstance(exc, TimeoutError):
        return "provider_timeout"
    if isinstance(exc, ValueError):
        return "invalid_plan_draft"
    return "provider_error"

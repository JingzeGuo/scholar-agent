"""Planner outputs: entities, answer requirements, and query plans."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from scholar_agent.models.base import QueryType


class SubQuestionStatus(StrEnum):
    PENDING = "pending"
    COVERED = "covered"
    MISSING = "missing"


class PlannedEntity(BaseModel):
    """An entity named by the user and resolved to a stable canonical identity."""

    id: str
    surface_name: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)

    @field_validator("id", "surface_name", "canonical_name")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @field_validator("aliases")
    @classmethod
    def _clean_aliases(cls, value: list[str]) -> list[str]:
        aliases: list[str] = []
        seen: set[str] = set()
        for alias in value:
            cleaned = alias.strip()
            folded = cleaned.casefold()
            if cleaned and folded not in seen:
                aliases.append(cleaned)
                seen.add(folded)
        return aliases


class AnswerRequirement(BaseModel):
    """A named answer dimension and the entities it must cover."""

    key: str
    description: str
    target_entity_ids: list[str] = Field(default_factory=list)

    @field_validator("key", "description")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @field_validator("target_entity_ids")
    @classmethod
    def _unique_target_entity_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("target_entity_ids must be unique")
        return value


class SubQuestion(BaseModel):
    id: str
    question: str
    query_type: QueryType
    required_evidence: list[str] = Field(default_factory=list)
    status: SubQuestionStatus = SubQuestionStatus.PENDING
    target_entity_ids: list[str] = Field(default_factory=list)
    requirement_keys: list[str] = Field(default_factory=list)
    dimension: str | None = None

    @field_validator("id", "question")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @field_validator("target_entity_ids", "requirement_keys")
    @classmethod
    def _unique_references(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("references must be unique")
        return value

    @field_validator("dimension")
    @classmethod
    def _clean_dimension(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class QueryPlan(BaseModel):
    original_query: str
    answer_type: str
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    expected_source_diversity: int = Field(default=1, ge=1)
    target_entities: list[PlannedEntity] = Field(default_factory=list)
    answer_requirements: list[AnswerRequirement] = Field(default_factory=list)

    @field_validator("original_query", "answer_type")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @field_validator("sub_questions")
    @classmethod
    def _unique_ids(cls, value: list[SubQuestion]) -> list[SubQuestion]:
        ids = [sq.id for sq in value]
        if len(ids) != len(set(ids)):
            raise ValueError("sub_question ids must be unique")
        return value

    @field_validator("target_entities")
    @classmethod
    def _unique_entity_ids(cls, value: list[PlannedEntity]) -> list[PlannedEntity]:
        ids = [entity.id for entity in value]
        if len(ids) != len(set(ids)):
            raise ValueError("target entity ids must be unique")
        return value

    @field_validator("answer_requirements")
    @classmethod
    def _unique_requirement_keys(
        cls,
        value: list[AnswerRequirement],
    ) -> list[AnswerRequirement]:
        keys = [requirement.key for requirement in value]
        if len(keys) != len(set(keys)):
            raise ValueError("answer requirement keys must be unique")
        return value

    @model_validator(mode="after")
    def _references_exist(self) -> QueryPlan:
        entity_ids = {entity.id for entity in self.target_entities}
        requirement_keys = {requirement.key for requirement in self.answer_requirements}
        for requirement in self.answer_requirements:
            unknown_entities = set(requirement.target_entity_ids) - entity_ids
            if unknown_entities:
                raise ValueError(
                    "answer requirement references unknown target entities: "
                    f"{sorted(unknown_entities)}"
                )
        for sub_question in self.sub_questions:
            unknown_entities = set(sub_question.target_entity_ids) - entity_ids
            if unknown_entities:
                raise ValueError(
                    "sub-question references unknown target entities: "
                    f"{sorted(unknown_entities)}"
                )
            unknown_requirements = set(sub_question.requirement_keys) - requirement_keys
            if unknown_requirements:
                raise ValueError(
                    "sub-question references unknown answer requirements: "
                    f"{sorted(unknown_requirements)}"
                )
        return self


class PlanDraftSubQuestion(BaseModel):
    """LLM-authored sub-question without runtime IDs."""

    question: str
    query_type: QueryType
    target_entities: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    dimension: str | None = None
    required_evidence: list[str] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def _question_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must be a non-empty string")
        return cleaned


class PlanDraft(BaseModel):
    """Constrained LLM output; stable identities are added by :class:`Planner`."""

    answer_type: str
    target_entities: list[str] = Field(default_factory=list)
    answer_requirements: list[str] = Field(default_factory=list)
    sub_questions: list[PlanDraftSubQuestion] = Field(default_factory=list)
    expected_source_diversity: int = Field(default=1, ge=1)

    @field_validator("answer_type")
    @classmethod
    def _answer_type_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("answer_type must be a non-empty string")
        return cleaned

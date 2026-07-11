"""Planner outputs: sub-questions and query plans."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from scholar_agent.models.base import QueryType


class SubQuestionStatus(StrEnum):
    PENDING = "pending"
    COVERED = "covered"
    MISSING = "missing"


class SubQuestion(BaseModel):
    id: str
    question: str
    query_type: QueryType
    required_evidence: list[str] = Field(default_factory=list)
    status: SubQuestionStatus = SubQuestionStatus.PENDING

    @field_validator("id", "question")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned


class QueryPlan(BaseModel):
    original_query: str
    answer_type: str
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    expected_source_diversity: int = Field(default=1, ge=1)

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

"""Tests for structured Planner."""

from __future__ import annotations

import json

import pytest

from scholar_agent.agents.planner import Planner, PlannerLLMError, extract_answer_anchors
from scholar_agent.llm.client import ChatResponse
from scholar_agent.llm.structured import StructuredOutputError, StructuredOutputErrorCode
from scholar_agent.models.base import QueryType, TokenUsage
from scholar_agent.models.planning import QueryPlan


class FakePlannerLLM:
    def __init__(
        self,
        *,
        content: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.error = error
        self.calls = 0
        self.messages: list[object] = []

    def chat_json(self, *args: object, **_kwargs: object) -> ChatResponse:
        self.calls += 1
        if args:
            messages = args[0]
            if isinstance(messages, list):
                self.messages = messages
        if self.error is not None:
            raise self.error
        return ChatResponse(
            content=self.content,
            model="fake-fast-planner",
            usage=TokenUsage(prompt_tokens=23, completion_tokens=17, total_tokens=40),
            raw={"private_provider_response": "must-not-be-retained"},
        )


def _valid_comparison_draft() -> str:
    return json.dumps(
        {
            "answer_type": "comparison",
            "target_entities": ["Self-RAG", "CRAG"],
            "answer_requirements": [
                "retrieval trigger",
                "correction mechanism",
                "key differences",
            ],
            "sub_questions": [
                {
                    "question": "How does Self-RAG trigger retrieval?",
                    "query_type": "comparison",
                    "target_entities": ["Self-RAG"],
                    "requirements": ["retrieval trigger"],
                    "dimension": "retrieval trigger",
                    "required_evidence": ["Self-RAG retrieval trigger"],
                },
                {
                    "question": "How does CRAG trigger retrieval?",
                    "query_type": "comparison",
                    "target_entities": ["CRAG"],
                    "requirements": ["retrieval trigger"],
                    "dimension": "retrieval trigger",
                    "required_evidence": ["CRAG retrieval trigger"],
                },
                {
                    "question": "How does Self-RAG correct weak evidence?",
                    "query_type": "comparison",
                    "target_entities": ["Self-RAG"],
                    "requirements": ["correction mechanism"],
                    "dimension": "correction mechanism",
                    "required_evidence": ["Self-RAG correction mechanism"],
                },
                {
                    "question": "How does CRAG correct weak evidence?",
                    "query_type": "comparison",
                    "target_entities": ["CRAG"],
                    "requirements": ["correction mechanism"],
                    "dimension": "correction mechanism",
                    "required_evidence": ["CRAG correction mechanism"],
                },
                {
                    "question": "What are their key differences?",
                    "query_type": "comparison",
                    "target_entities": ["Self-RAG", "CRAG"],
                    "requirements": ["key differences"],
                    "dimension": "key differences",
                    "required_evidence": ["both methods"],
                },
            ],
            "expected_source_diversity": 2,
        }
    )


def _comparison_draft_with_substituted_entity() -> str:
    payload = json.loads(_valid_comparison_draft())
    payload["target_entities"] = ["Self-RAG", "GraphRAG"]
    for sub_question in payload["sub_questions"]:
        sub_question["target_entities"] = [
            "GraphRAG" if entity == "CRAG" else entity for entity in sub_question["target_entities"]
        ]
    return json.dumps(payload)


def _single_entity_draft(entity: str) -> str:
    return json.dumps(
        {
            "answer_type": "semantic",
            "target_entities": [entity],
            "answer_requirements": ["definition"],
            "sub_questions": [
                {
                    "question": f"What is {entity}?",
                    "query_type": "semantic",
                    "target_entities": [entity],
                    "requirements": ["definition"],
                    "dimension": "definition",
                    "required_evidence": ["definition", entity],
                }
            ],
            "expected_source_diversity": 1,
        }
    )


def test_simple_factual_not_over_decomposed() -> None:
    plan = Planner().plan("What is RAPTOR?")
    assert len(plan.sub_questions) == 1
    assert plan.sub_questions[0].query_type in {QueryType.SEMANTIC, QueryType.KEYWORD}
    assert plan.original_query == "What is RAPTOR?"


def test_comparison_produces_multiple_subquestions() -> None:
    plan = Planner().plan("Compare Self-RAG versus CRAG")
    assert plan.answer_type == "comparison"
    assert len(plan.sub_questions) >= 2
    assert plan.expected_source_diversity >= 2
    ids = [sq.id for sq in plan.sub_questions]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    "query",
    [
        "Self-RAG vs CRAG",
        "Self-RAG vs. CRAG",
        "Compare Self-RAG versus CRAG",
        "Compare Self-RAG and CRAG",
        "What are the differences between Self-RAG and CRAG?",
    ],
)
def test_comparison_syntax_resolves_two_canonical_entities(query: str) -> None:
    plan = Planner().plan(query)
    assert [entity.canonical_name for entity in plan.target_entities] == [
        "Self-RAG",
        "Corrective RAG",
    ]
    assert plan.target_entities[1].surface_name == "CRAG"
    assert "CRAG" in plan.target_entities[1].aliases
    assert all(sub_question.target_entity_ids for sub_question in plan.sub_questions)


def test_multisentence_comparison_extracts_entities_and_requested_dimensions() -> None:
    query = (
        "Compare Self-RAG versus CRAG. "
        "Explain their retrieval triggers, correction mechanisms, and key differences."
    )
    plan = Planner().plan(query)

    assert [entity.canonical_name for entity in plan.target_entities] == [
        "Self-RAG",
        "Corrective RAG",
    ]
    assert [requirement.key for requirement in plan.answer_requirements] == [
        "retrieval_trigger",
        "correction_mechanism",
        "key_differences",
    ]
    assert all(
        requirement.target_entity_ids == [entity.id for entity in plan.target_entities]
        for requirement in plan.answer_requirements
    )
    assert len(plan.sub_questions) == 4
    assert all("CRAG. Explain" not in sub_question.question for sub_question in plan.sub_questions)
    assert {
        (sub_question.dimension, tuple(sub_question.target_entity_ids))
        for sub_question in plan.sub_questions
    } == {
        ("retrieval_trigger", (plan.target_entities[0].id,)),
        ("retrieval_trigger", (plan.target_entities[1].id,)),
        ("correction_mechanism", (plan.target_entities[0].id,)),
        ("correction_mechanism", (plan.target_entities[1].id,)),
    }
    assert all(
        "key_differences" not in sub_question.requirement_keys
        for sub_question in plan.sub_questions
    )
    assert all(
        sub_question.query_type == QueryType.COMPARISON for sub_question in plan.sub_questions
    )

    repeated = Planner().plan(query)
    assert [entity.id for entity in repeated.target_entities] == [
        entity.id for entity in plan.target_entities
    ]
    assert [sub_question.id for sub_question in repeated.sub_questions] == [
        sub_question.id for sub_question in plan.sub_questions
    ]


def test_comprehensive_rag_benchmark_is_not_resolved_as_corrective_rag() -> None:
    plan = Planner().plan("Compare Self-RAG versus CRAG -- Comprehensive RAG Benchmark.")
    assert [entity.canonical_name for entity in plan.target_entities] == [
        "Self-RAG",
        "Comprehensive RAG Benchmark",
    ]


def test_synthesis_plan() -> None:
    plan = Planner().plan("Summarize main trends across agentic RAG papers")
    assert plan.answer_type == "synthesis"
    assert len(plan.sub_questions) >= 2


def test_plan_is_structured_not_string() -> None:
    plan = Planner().plan("Which datasets does DPR evaluate on?")
    assert hasattr(plan, "sub_questions")
    assert all(hasattr(sq, "required_evidence") for sq in plan.sub_questions)


def test_named_versions_and_years_become_exact_answer_anchors() -> None:
    query = "Which GPU powered AcmeAI's private 2027 training run for Model-X2?"
    assert extract_answer_anchors(query) == ["GPU", "AcmeAI", "2027", "Model-X2"]
    plan = Planner().plan(query)
    requirements = plan.sub_questions[0].required_evidence
    assert "anchor:GPU" in requirements
    assert "anchor:AcmeAI" in requirements
    assert "anchor:2027" in requirements
    assert "anchor:Model-X2" in requirements


def test_old_query_plan_payload_remains_compatible() -> None:
    plan = QueryPlan.model_validate(
        {
            "original_query": "What is RAPTOR?",
            "answer_type": "factual",
            "sub_questions": [
                {
                    "id": "sq_legacy",
                    "question": "What is RAPTOR?",
                    "query_type": "keyword",
                }
            ],
        }
    )
    assert plan.target_entities == []
    assert plan.answer_requirements == []
    assert plan.sub_questions[0].target_entity_ids == []
    assert plan.sub_questions[0].requirement_keys == []
    assert plan.sub_questions[0].dimension is None


def test_llm_plan_draft_is_validated_and_ids_are_generated_locally() -> None:
    fake = FakePlannerLLM(content=_valid_comparison_draft())
    planner = Planner(llm=fake)  # type: ignore[arg-type]
    plan = planner.plan("Compare Self-RAG and CRAG")

    assert fake.calls == 1
    assert planner.last_backend == "llm"
    assert planner.last_model == "fake-fast-planner"
    assert planner.last_fallback_reason is None
    assert planner.last_token_usage.total_tokens == 40
    assert planner.last_prompt_version == "planner-plan-draft-v2"
    assert all(entity.id.startswith("ent_") for entity in plan.target_entities)
    assert all(sub_question.id.startswith("sq_") for sub_question in plan.sub_questions)
    assert [entity.canonical_name for entity in plan.target_entities] == [
        "Self-RAG",
        "Corrective RAG",
    ]
    assert len(plan.sub_questions) == 4
    assert all(
        sub_question.query_type == QueryType.COMPARISON
        and "key_differences" not in sub_question.requirement_keys
        for sub_question in plan.sub_questions
    )
    system_prompt = str(getattr(fake.messages[0], "content", ""))
    assert '"sub_questions"' in system_prompt
    assert '"expected_source_diversity"' in system_prompt
    assert "All keys shown in the output contract are required" in system_prompt
    assert "private_provider_response" not in str(planner.__dict__)


def test_malformed_llm_output_falls_back_without_retaining_raw_response() -> None:
    fake = FakePlannerLLM(content="provider-private malformed response")
    planner = Planner(llm=fake)  # type: ignore[arg-type]
    plan = planner.plan("Compare Self-RAG and CRAG")

    assert plan.answer_type == "comparison"
    assert planner.last_backend == "deterministic"
    assert planner.last_model == "fake-fast-planner"
    assert planner.last_fallback_reason == "json_decode_failed"
    assert planner.last_token_usage.total_tokens == 40
    assert "provider-private malformed response" not in str(planner.__dict__)


def test_llm_missing_field_has_safe_classification_and_path() -> None:
    payload = json.loads(_valid_comparison_draft())
    del payload["sub_questions"][0]["query_type"]
    private_value = "private-provider-field-value"
    payload["private_extra"] = private_value
    planner = Planner(
        llm=FakePlannerLLM(content=json.dumps(payload)),
        strict_llm=True,
    )  # type: ignore[arg-type]

    with pytest.raises(PlannerLLMError) as exc_info:
        planner.plan("Compare Self-RAG and CRAG")

    cause = exc_info.value.__cause__
    assert isinstance(cause, StructuredOutputError)
    assert cause.code == StructuredOutputErrorCode.MISSING_REQUIRED_FIELD
    assert cause.field_paths == ("sub_questions[0].query_type",)
    assert private_value not in str(cause)
    assert planner.last_fallback_reason == "missing_required_field"
    assert planner.last_fallback_fields == ("sub_questions[0].query_type",)


def test_llm_unknown_entity_reference_has_safe_path() -> None:
    payload = json.loads(_valid_comparison_draft())
    payload["sub_questions"][0]["target_entities"] = ["GraphRAG"]
    planner = Planner(
        llm=FakePlannerLLM(content=json.dumps(payload)),
        strict_llm=True,
    )  # type: ignore[arg-type]

    with pytest.raises(PlannerLLMError) as exc_info:
        planner.plan("Compare Self-RAG and CRAG")

    cause = exc_info.value.__cause__
    assert isinstance(cause, StructuredOutputError)
    assert cause.code == StructuredOutputErrorCode.UNKNOWN_ENTITY_ID
    assert cause.field_paths == ("sub_questions[0].target_entities[0]",)
    assert planner.last_fallback_reason == "unknown_entity_id"


def test_llm_unknown_requirement_reference_has_safe_path() -> None:
    payload = json.loads(_valid_comparison_draft())
    payload["sub_questions"][0]["requirements"] = ["private invented dimension"]
    planner = Planner(
        llm=FakePlannerLLM(content=json.dumps(payload)),
        strict_llm=True,
    )  # type: ignore[arg-type]

    with pytest.raises(PlannerLLMError) as exc_info:
        planner.plan("Compare Self-RAG and CRAG")

    cause = exc_info.value.__cause__
    assert isinstance(cause, StructuredOutputError)
    assert cause.code == StructuredOutputErrorCode.UNKNOWN_REQUIREMENT_KEY
    assert cause.field_paths == ("sub_questions[0].requirements[0]",)
    assert "private invented dimension" not in str(cause)
    assert planner.last_fallback_reason == "unknown_requirement_key"


def test_llm_derived_difference_requirement_needs_no_research_subquestion() -> None:
    payload = json.loads(_valid_comparison_draft())
    payload["sub_questions"] = [
        sub_question
        for sub_question in payload["sub_questions"]
        if sub_question["dimension"] != "key differences"
    ]
    planner = Planner(llm=FakePlannerLLM(content=json.dumps(payload)))  # type: ignore[arg-type]

    plan = planner.plan("Compare Self-RAG and CRAG")

    assert planner.last_backend == "llm"
    assert planner.last_fallback_reason is None
    assert {requirement.key for requirement in plan.answer_requirements} == {
        "retrieval_trigger",
        "correction_mechanism",
        "key_differences",
    }


def test_llm_provider_exception_falls_back_deterministically() -> None:
    fake = FakePlannerLLM(error=TimeoutError("private provider details"))
    planner = Planner(llm=fake)  # type: ignore[arg-type]
    plan = planner.plan("Compare Self-RAG and CRAG")

    assert plan.target_entities
    assert planner.last_backend == "deterministic"
    assert planner.last_fallback_reason == "provider_timeout"
    assert "private provider details" not in str(planner.__dict__)


def test_llm_comparison_entity_substitution_falls_back_to_query_entities() -> None:
    fake = FakePlannerLLM(content=_comparison_draft_with_substituted_entity())
    planner = Planner(llm=fake)  # type: ignore[arg-type]
    plan = planner.plan("Compare Self-RAG and CRAG")

    assert [entity.canonical_name for entity in plan.target_entities] == [
        "Self-RAG",
        "Corrective RAG",
    ]
    assert all(entity.canonical_name != "GraphRAG" for entity in plan.target_entities)
    assert planner.last_backend == "deterministic"
    assert planner.last_fallback_reason == "unknown_entity_id"


def test_llm_noncomparison_entity_requires_query_boundary_anchor() -> None:
    fake = FakePlannerLLM(content=_single_entity_draft("GraphRAG"))
    planner = Planner(llm=fake)  # type: ignore[arg-type]
    plan = planner.plan("What is CRAG?")

    assert [entity.canonical_name for entity in plan.target_entities] == ["Corrective RAG"]
    assert planner.last_backend == "deterministic"
    assert planner.last_fallback_reason == "unknown_entity_id"


def test_llm_known_entity_does_not_match_inside_more_specific_alias() -> None:
    fake = FakePlannerLLM(content=_single_entity_draft("RAG"))
    planner = Planner(llm=fake)  # type: ignore[arg-type]
    plan = planner.plan("What is GraphRAG?")

    assert [entity.canonical_name for entity in plan.target_entities] == ["GraphRAG"]
    assert planner.last_backend == "deterministic"
    assert planner.last_fallback_reason == "unknown_entity_id"


def test_strict_llm_rejects_comparison_entity_substitution() -> None:
    fake = FakePlannerLLM(content=_comparison_draft_with_substituted_entity())
    planner = Planner(llm=fake, strict_llm=True)  # type: ignore[arg-type]

    with pytest.raises(PlannerLLMError, match="unknown_entity_id"):
        planner.plan("Compare Self-RAG and CRAG")
    assert planner.last_backend == "llm"
    assert planner.last_fallback_reason == "unknown_entity_id"


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("not-json", None),
        (None, RuntimeError("provider failed")),
    ],
)
def test_strict_llm_mode_never_falls_back(
    content: str | None,
    error: Exception | None,
) -> None:
    fake = FakePlannerLLM(content=content, error=error)
    planner = Planner(llm=fake, strict_llm=True)  # type: ignore[arg-type]
    with pytest.raises(PlannerLLMError, match="LLM planner failed"):
        planner.plan("Compare Self-RAG and CRAG")
    assert planner.last_backend == "llm"
    assert planner.last_fallback_reason is not None

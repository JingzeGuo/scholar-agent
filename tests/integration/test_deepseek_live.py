"""Optional live DeepSeek tests. Run with: pytest -m live

These tests are skipped unless DEEPSEEK_API_KEY (or OPENAI_API_KEY) is set.
"""

from __future__ import annotations

import os

import pytest
from scripts.deepseek_compatibility import (
    check_streaming,
    check_structured_json,
    check_tool_calling,
)

from scholar_agent.agents.planner import Planner
from scholar_agent.agents.writer import Writer
from scholar_agent.config import load_config
from scholar_agent.llm.client import ChatMessage, create_llm_client
from scholar_agent.models.answer import AnswerStatus
from scholar_agent.models.base import QueryType
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger

pytestmark = pytest.mark.live


def _has_api_key() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))


@pytest.mark.skipif(not _has_api_key(), reason="No live API key configured")
def test_live_chat_completion(repo_root: object) -> None:
    from pathlib import Path

    root = Path(str(repo_root))
    config = load_config(root / "configs" / "default.yaml", repo_root=root)
    client = create_llm_client(config)
    response = client.chat(
        [ChatMessage(role="user", content="Reply with the single word: pong")],
        fast=True,
        max_tokens=16,
    )
    assert response.content
    assert response.content.strip()


@pytest.mark.skipif(not _has_api_key(), reason="No live API key configured")
def test_live_phase_zero_acceptance(repo_root: object) -> None:
    from pathlib import Path

    root = Path(str(repo_root))
    config = load_config(root / "configs" / "default.yaml", repo_root=root)
    client = create_llm_client(config)
    model = config.llm.fast_model

    results = [
        check_streaming(client, model),
        check_structured_json(client, model),
        check_tool_calling(client, model),
    ]
    failures = [f"{result.name}: {result.detail}" for result in results if not result.passed]
    assert not failures, "; ".join(failures)


@pytest.mark.skipif(not _has_api_key(), reason="No live API key configured")
def test_live_phase_eleven_structured_planner_and_writer(repo_root: object) -> None:
    """Exercise the real strict Planner and Writer paths without retrieval I/O."""
    from pathlib import Path

    root = Path(str(repo_root))
    config = load_config(root / "configs" / "default.yaml", repo_root=root)
    client = create_llm_client(config)
    query = (
        "Compare Self-RAG versus CRAG. "
        "Explain their retrieval triggers, correction mechanisms, and key differences."
    )

    planner = Planner(llm=client, strict_llm=True)
    plan = planner.plan(query)

    assert planner.last_backend == "llm"
    assert planner.last_fallback_reason is None
    assert len(plan.sub_questions) == 4
    assert all(
        sub_question.dimension in {"retrieval_trigger", "correction_mechanism"}
        and sub_question.query_type == QueryType.COMPARISON
        and len(sub_question.target_entity_ids) == 1
        and len(sub_question.requirement_keys) == 1
        and "key_differences" not in sub_question.requirement_keys
        for sub_question in plan.sub_questions
    )

    entity_ids = {entity.canonical_name: entity.id for entity in plan.target_entities}
    expected_pairs = {
        (entity_ids["Self-RAG"], "retrieval_trigger"),
        (entity_ids["Self-RAG"], "correction_mechanism"),
        (entity_ids["Corrective RAG"], "retrieval_trigger"),
        (entity_ids["Corrective RAG"], "correction_mechanism"),
    }
    sub_questions_by_pair = {
        (sub_question.target_entity_ids[0], sub_question.requirement_keys[0]): sub_question
        for sub_question in plan.sub_questions
    }
    assert set(sub_questions_by_pair) == expected_pairs

    evidence_specs = [
        (
            entity_ids["Self-RAG"],
            "retrieval_trigger",
            "paper_arxiv_2310_11511",
            "Self-RAG uses reflection tokens to decide when to retrieve passages on demand.",
        ),
        (
            entity_ids["Self-RAG"],
            "correction_mechanism",
            "paper_arxiv_2310_11511",
            "Self-RAG uses critique reflection tokens to evaluate and refine its response.",
        ),
        (
            entity_ids["Corrective RAG"],
            "retrieval_trigger",
            "paper_arxiv_2401_15884",
            (
                "CRAG uses a retrieval evaluator to classify retrieved documents as "
                "Correct, Ambiguous, or Incorrect."
            ),
        ),
        (
            entity_ids["Corrective RAG"],
            "correction_mechanism",
            "paper_arxiv_2401_15884",
            (
                "CRAG corrects weak retrieval by refining relevant knowledge and "
                "invoking web search when needed."
            ),
        ),
    ]
    ledger = EvidenceLedger(
        items=[
            EvidenceItem(
                evidence_id=f"ev_live_{index}",
                sub_question_id=sub_questions_by_pair[(entity_id, requirement_key)].id,
                claim=text,
                evidence_text=text,
                paper_id=paper_id,
                chunk_id=f"chunk_live_{index}",
                page_start=index,
                page_end=index,
                retrieval_method="live_fixture",
                support_score=1.0,
            )
            for index, (entity_id, requirement_key, paper_id, text) in enumerate(
                evidence_specs,
                start=1,
            )
        ]
    )

    writer = Writer(llm=client, strict_llm=True)
    draft = writer.write(query=query, plan=plan, ledger=ledger)

    assert writer.last_backend == "llm"
    assert writer.last_fallback_reason is None
    assert draft.status == AnswerStatus.COMPLETE

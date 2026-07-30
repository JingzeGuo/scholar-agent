"""Writer Agent node."""

from __future__ import annotations

import logging
import re

from scholar_agent.citations import (
    citation_summary,
    valid_evidence_ids,
    validate_citations,
)
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)


def _first_sentence(text: str, limit: int = 260) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    if len(sentence) <= limit:
        return sentence
    return sentence[: limit - 1].rstrip() + "…"


def _offline_draft(state: AgentState) -> str:
    if not state["evidence"]:
        return "The available evidence is insufficient to answer this question."
    lead = (
        "The available evidence is limited, so the comparison is tentative:"
        if not state["sufficient"]
        else "The retrieved evidence supports this comparison:"
    )
    points = [
        f"- {_first_sentence(item['text'])} [E{index}]"
        for index, item in enumerate(state["evidence"][:4], start=1)
    ]
    return "\n".join([lead, *points])


def _writer_prompt(state: AgentState) -> str:
    evidence_text = "\n".join(
        f"[E{index}] {item['text']}"
        for index, item in enumerate(state["evidence"], start=1)
    )
    sufficiency = "sufficient" if state["sufficient"] else "incomplete"
    return f"""You are the Writer in an evidence-grounded research workflow.

Answer the question using only the evidence below. Every factual statement
must be supported by an inline [E1], [E2], ... reference. Never cite an ID that
is not supplied. Do not use background knowledge. When evidence is incomplete,
say exactly what remains uncertain.

Verification status: {sufficiency}

Question:
{state["question"]}

Evidence:
{evidence_text}
"""


def writer_node(state: AgentState, llm: LLMClient | None = None) -> dict:
    """Draft only from evidence, then validate every evidence reference."""
    if llm is None or not state["evidence"]:
        draft = _offline_draft(state)
    else:
        draft = llm.complete(_writer_prompt(state))
        if not valid_evidence_ids(draft, len(state["evidence"])):
            LOGGER.warning("[writer] LLM draft had no valid evidence IDs; using fallback")
            draft = _offline_draft(state)
    answer = validate_citations(draft, state["evidence"])
    summary = citation_summary(answer, state["evidence"])
    LOGGER.info(
        "[writer] answer generated with %d citations from %d sources",
        summary["citations"],
        summary["sources"],
    )
    return {"answer": answer}

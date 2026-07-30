"""Verifier Agent node."""

from __future__ import annotations

import logging

from scholar_agent.indexes import tokenize
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)


def _deterministic_verification(state: AgentState) -> tuple[bool, str]:
    evidence = state["evidence"]
    if not evidence:
        return False, "Find directly relevant evidence for the question."

    question_terms = {term for term in tokenize(state["question"]) if len(term) > 2}
    evidence_terms = set(tokenize(" ".join(item["text"] for item in evidence)))
    covered = question_terms.intersection(evidence_terms)
    coverage = len(covered) / max(len(question_terms), 1)
    asks_comparison = any(word in question_terms for word in {"compare", "versus", "difference"})
    papers = {item["paper"] for item in evidence}
    sufficient = coverage >= 0.45 and (not asks_comparison or len(papers) >= 2)
    if sufficient:
        return True, ""
    missing = sorted(question_terms - evidence_terms)[:5]
    detail = ", ".join(missing) if missing else "another independent source"
    return False, f"Retrieve missing comparison evidence about: {detail}."


def _verifier_prompt(state: AgentState) -> str:
    evidence_text = "\n".join(
        f"E{index}: {item['text']}"
        for index, item in enumerate(state["evidence"], start=1)
    )
    return f"""You are the Verifier in an academic research workflow.

Judge only whether the supplied evidence can answer the question. Do not add
facts, rewrite the answer, or propose more than the missing information.

Return one JSON object:
- "sufficient": true only when the evidence covers the requested comparison
- "feedback": empty when sufficient; otherwise one concise retrieval request

Question:
{state["question"]}

Evidence:
{evidence_text}
"""


def verifier_node(state: AgentState, llm: LLMClient | None = None) -> dict:
    """Judge sufficiency and state only the missing evidence."""
    sufficient, feedback = _deterministic_verification(state)
    if llm is not None:
        try:
            payload = llm.complete_json(_verifier_prompt(state))
            sufficient = bool(payload["sufficient"])
            feedback = str(payload.get("feedback", "")).strip()
        except (KeyError, TypeError, ValueError):
            pass
    if not state["evidence"]:
        sufficient = False
        feedback = feedback or "Find directly relevant evidence for the question."

    LOGGER.info(
        "[verifier] evidence %s",
        "sufficient" if sufficient else "insufficient",
    )
    return {
        "sufficient": sufficient,
        "feedback": feedback,
        "retry_count": state["retry_count"],
    }

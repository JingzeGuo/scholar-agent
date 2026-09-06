"""LLM-based grounded-answer node."""

from __future__ import annotations

import json
import logging

from scholar_agent.citations import (
    PAGE_CITATION_RE,
    citation_summary,
    validate_citations,
)
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
SAFE_ABSTENTION = "The supplied evidence is insufficient to provide a citation-grounded answer."


def _writer_prompt(state: AgentState) -> str:
    evidence_text = "\n".join(
        f"[E{index}] {item['text']}"
        for index, item in enumerate(state["evidence"], start=1)
    )
    verification = state["verification"]
    return f"""You are the Writer in an evidence-grounded research workflow.

Answer in English using only the supplied evidence.
Every factual statement, including an opening summary or concluding restatement, needs an
inline supplied [E1], [E2], ... reference.
For multiple sources, write adjacent references like [E1][E5].
Use only citations that directly support the sentence, and repeat a citation whenever another
factual sentence requires it.
Do not use evidence IDs that were not supplied.
Do not substitute related methods for explicitly named targets.
Respect constraints in the original question only when supported by evidence.
Answer only supported aspects and do not fill missing gaps from memory.
Organize the answer around the user's question rather than around evidence chunks.
Treat the coverage analysis as advisory. Inspect all evidence yourself, and do not claim that
an item is missing when any supplied evidence supports it.

If no supplied evidence supports any requested factual answer, return exactly:
{SAFE_ABSTENTION}

Coverage status: {verification["status"]}
Requirements: {state["plan"]["requirements"]}
Covered: {verification["covered"]}
Uncertain: {verification.get("uncertain", {})}
Missing: {verification["missing"]}
Question: {state["question"]}

Evidence:
{evidence_text}
"""


def _render_answer(draft: str, evidence: list[dict]) -> str:
    return validate_citations(PAGE_CITATION_RE.sub("", draft), evidence)


def writer_node(state: AgentState, llm: LLMClient) -> dict:
    """Write from all selected evidence, using coverage only as advice."""
    status = state["verification"]["status"]
    if not state["evidence"]:
        LOGGER.info("[writer] deterministic abstention without evidence")
        return {"answer": SAFE_ABSTENTION}

    answer = _render_answer(llm.complete(_writer_prompt(state)), state["evidence"])
    summary = citation_summary(answer, state["evidence"])
    LOGGER.info(
        "[writer] status=%s citations=%d sources=%d",
        status,
        summary["citations"],
        summary["sources"],
    )
    return {"answer": answer}


def repair_writer_node(state: AgentState, llm: LLMClient) -> dict:
    """Repair only the issues reported by the answer verifier, at most once."""
    evidence_text = "\n".join(
        f"[E{index}] {item['text']}"
        for index, item in enumerate(state["evidence"], start=1)
    )
    issues = json.dumps(
        state["answer_verification"],
        ensure_ascii=False,
        indent=2,
    )
    prompt = f"""Repair an evidence-grounded academic answer in English.

Use only the supplied evidence and change only what the verification issues require.
Every factual sentence needs a directly supporting [E1], [E2], ... citation.
Do not add new claims. Do not claim that evidence is missing when it is supplied.

Question: {state["question"]}
Requirements: {state["plan"]["requirements"]}

Verification issues:
{issues}

Previous answer:
{state["answer"]}

Evidence:
{evidence_text}
"""
    answer = _render_answer(llm.complete(prompt), state["evidence"])
    return {"answer": answer, "repair_count": state["repair_count"] + 1}

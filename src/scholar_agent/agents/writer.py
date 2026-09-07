"""LLM-based grounded-answer node."""

from __future__ import annotations

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
    return f"""You are the Writer in an evidence-grounded research workflow.

Answer in English using only the supplied evidence.
Output only directly supported answer sentences. Start with the first supported claim and its
citation; do not add an introductory overview, thesis sentence, or uncited opening summary.
Every sentence that identifies, describes, compares, or concludes something factual must contain
at least one directly supporting [E1], [E2], ... reference immediately before its punctuation.
This rule also applies to short opening sentences, bullet items, transitions, and restatements.
Do not end with a summary or conclusion that repeats factual claims. End after the last supported,
cited detail. If a concluding sentence is essential, cite that sentence independently.
For multiple sources, write adjacent references like [E1][E5].
Use only citations that directly support the sentence, and repeat a citation whenever another
factual sentence requires it.
Do not use evidence IDs that were not supplied.
Do not substitute related methods for explicitly named targets.
Respect constraints in the original question only when supported by evidence.
Answer only supported aspects and do not fill missing gaps from memory.
Organize the answer around the user's question rather than around evidence chunks.
Inspect all evidence yourself, and do not claim that an item is missing when any supplied
evidence supports it.
Before returning, inspect every sentence: delete any factual sentence that lacks its own adjacent
evidence reference. A citation in a neighboring sentence never supports an uncited sentence.

If no supplied evidence supports any requested factual answer, return exactly:
{SAFE_ABSTENTION}

Requirements: {state["plan"]["requirements"]}
Question: {state["question"]}

Evidence:
{evidence_text}
"""


def writer_node(state: AgentState, llm: LLMClient) -> dict:
    """Write one grounded draft from all selected evidence."""
    if not state["evidence"]:
        LOGGER.info("[writer] deterministic abstention without evidence")
        return {"answer": SAFE_ABSTENTION}

    answer = llm.complete(_writer_prompt(state)).strip()
    LOGGER.info("[writer] produced draft")
    return {"answer": answer}


def citation_validator_node(state: AgentState) -> dict:
    """Render known evidence IDs and remove fabricated page citations deterministically."""
    answer = validate_citations(
        PAGE_CITATION_RE.sub("", state["answer"]),
        state["evidence"],
    )
    summary = citation_summary(answer, state["evidence"])
    LOGGER.info(
        "[citations] citations=%d sources=%d",
        summary["citations"],
        summary["sources"],
    )
    return {"answer": answer}

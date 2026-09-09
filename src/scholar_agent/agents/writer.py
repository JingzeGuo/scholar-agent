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
    evidence_by_id = {item["id"]: item for item in state["evidence"]}

    def passage(evidence_id: str) -> str:
        item = evidence_by_id[evidence_id]
        source = item["paper"]
        if item.get("title"):
            source = f"{item['title']} ({source})"
        section = f" — {item['section']}" if item.get("section") else ""
        return f"[{evidence_id}] {source} — p.{item['page']}{section}\n{item['text']}"

    blocks = []
    assigned_ids: set[str] = set()
    for requirement_id, entry in state["evidence_board"].items():
        evidence_ids = entry["evidence_ids"]
        assigned_ids.update(evidence_ids)
        supporting = "\n\n".join(passage(evidence_id) for evidence_id in evidence_ids)
        blocks.append(
            f"Requirement {requirement_id}:\n{entry['requirement']}\n\nSupporting evidence:\n"
            + (supporting or "No matching evidence selected for this requirement."),
        )
    unassigned = [passage(evidence_id) for evidence_id in evidence_by_id if evidence_id not in assigned_ids]
    if unassigned:
        blocks.append("Additional selected evidence (no requirement match):\n" + "\n\n".join(unassigned))
    evidence_text = "\n\n".join(blocks)
    return f"""You are the Writer in an evidence-grounded research workflow.

Answer in English using only the supplied evidence.
Output only directly supported factual answers and brief evidence-gap statements.
Start with the first supported claim and its
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
Use the Requirement–Evidence Blackboard to address each requirement. Its evidence links are
retrieval relevance hints, not proof that a passage supports every part of the requirement.
Check the passage text before making a claim. The same evidence ID always identifies the same
passage, even when it appears under multiple requirements. You may use any supplied passage
that directly supports the claim, including evidence listed under another requirement.
When some requirements are supported, explicitly state which remaining requirements lack
support in the supplied evidence. These evidence-gap statements need no citation; do not
invent facts or evidence IDs to fill the gaps, and do not turn an empty board entry into a
claim that the information does not exist in the corpus or elsewhere.
Inspect all evidence yourself, and do not claim that an item is missing when any supplied
evidence supports it.
Before returning, inspect every sentence: delete any factual sentence that lacks its own adjacent
evidence reference. A citation in a neighboring sentence never supports an uncited sentence.

If no supplied evidence supports any requested factual answer, return exactly:
{SAFE_ABSTENTION}

Question: {state["question"]}

Requirement–Evidence Blackboard:
{evidence_text}
"""


def writer_node(state: AgentState, llm: LLMClient) -> dict:
    """Write one grounded draft from the requirement–evidence board."""
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

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


def _writer_context(state: AgentState, use_evidence_board: bool) -> str:
    evidence_by_id = {item["id"]: item for item in state["evidence"]}

    def passage(evidence_id: str) -> str:
        item = evidence_by_id[evidence_id]
        source = item["paper"]
        if item.get("title"):
            source = f"{item['title']} ({source})"
        section = f" — {item['section']}" if item.get("section") else ""
        return f"[{evidence_id}] {source} — p.{item['page']}{section}\n{item['text']}"

    def assessment(requirement_id: str) -> str:
        entry = state["evidence_board"].get(requirement_id, {})
        if entry.get("status", "unknown") == "unknown":
            return ""
        covered = "; ".join(entry.get("covered", [])) or "None identified"
        missing = "; ".join(entry.get("missing", [])) or "None"
        action = entry.get("action")
        recovery = ""
        if isinstance(action, dict) and action.get("state") == "executed":
            recovery = (
                "\nRecovery: executed after this assessment; check only whether the recovered "
                "candidate passages resolve the listed missing aspect."
            )
        return (
            "\nController coverage assessment:\n"
            f"Status: {entry['status']}\n"
            f"Covered: {covered}\n"
            f"Missing: {missing}{recovery}"
        )

    if not use_evidence_board:
        requirements = "\n\n".join(
            f"Requirement {item['id']}:\n{item['description']}{assessment(item['id'])}"
            for item in state["plan"]["requirements"]
        )
        evidence = "\n\n".join(passage(evidence_id) for evidence_id in evidence_by_id)
        return f"Requirements:\n{requirements}\n\nEvidence:\n{evidence}"

    blocks = []
    assigned_ids: set[str] = set()
    for requirement_id, entry in state["evidence_board"].items():
        evidence_ids = entry["evidence_ids"]
        assigned_ids.update(evidence_ids)
        supporting = "\n\n".join(passage(evidence_id) for evidence_id in evidence_ids)
        blocks.append(
            f"Requirement {requirement_id}:\n{entry['requirement']}"
            f"{assessment(requirement_id)}\n\nCandidate supporting evidence:\n"
            + (supporting or "No matching evidence selected for this requirement."),
        )
    unassigned = [passage(evidence_id) for evidence_id in evidence_by_id if evidence_id not in assigned_ids]
    if unassigned:
        blocks.append(
            "Additional candidate evidence (not linked to a requirement):\n"
            + "\n\n".join(unassigned),
        )
    return "Requirement–Evidence Blackboard:\n" + "\n\n".join(blocks)


def _writer_prompt(state: AgentState, *, use_evidence_board: bool = True) -> str:
    """Keep answer policy identical when ablating only the evidence layout."""
    context = _writer_context(state, use_evidence_board)
    length_policy = ""
    if state["plan"].get("answer_length") == "short":
        length_policy = """
This is a low-complexity definition question. Give the definition first, keep the entire answer
to 2–5 sentences, and do not broaden it into history, surveys, benchmarks, or adjacent methods.
"""
    return f"""You are the Writer in an evidence-grounded research workflow.

Answer in English using only the supplied evidence.
{length_policy}
Output only directly supported factual answers. Mention an evidence gap only if it prevents you
from answering an important part of the user's question; do not report peripheral gaps.
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
Requirements are research scaffolding, not an answer outline. Structure the final answer around
the original user question. Treat the supplied evidence as a candidate support pool and use only
the subset needed to answer clearly and directly. Do not mention a fact merely because supporting
evidence is available. Prefer the shortest answer that fully satisfies the user's intent.
If the question uses an acronym or named entity without disambiguating context and the candidate
evidence supports multiple identities, state that it is ambiguous and briefly distinguish the
relevant meanings. Do not select one identity merely because its passage has the highest score.
When a Controller coverage assessment is supplied, use it instead of redoing the initial coverage
analysis. If recovery ran afterward, check only whether the recovered candidates resolve its listed
gap. The status itself does not need to be mentioned. Check the cited passage before making each
factual claim: requirement–evidence links identify candidates but do not prove textual support. The
same evidence ID always identifies the same passage, and evidence linked elsewhere may be cited.
Evidence-gap statements need no citation. Do not invent facts or evidence IDs to fill a gap, and
do not turn missing support into a claim that the information does not exist in the corpus or
elsewhere.
Before returning, inspect every sentence: delete any factual sentence that lacks its own adjacent
evidence reference. A citation in a neighboring sentence never supports an uncited sentence.

If no supplied evidence supports any requested factual answer, return exactly:
{SAFE_ABSTENTION}

Question: {state["question"]}

{context}
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

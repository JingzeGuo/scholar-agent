"""LLM-based grounded-answer node."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from scholar_agent.citations import (
    PAGE_CITATION_RE,
    citation_summary,
    validate_citations,
)
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)
SAFE_ABSTENTION = "The supplied evidence is insufficient to provide a citation-grounded answer."
STREAM_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:[ \t]+|\n+)|\n+")


def _writer_context(state: AgentState) -> str:
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


def _render_writer_prompt(question: str, context: str) -> str:
    """Render the production answer policy around a prepared evidence context."""
    return f"""You are the Writer in an evidence-grounded research workflow.

Answer in English using only the supplied evidence.
Requirements are research scaffolding, not an answer outline. Structure the final answer around
the original question. Before writing, infer the smallest set of claims needed to satisfy it.
Use only the necessary subset of the candidate evidence. Add a sentence only when it answers a
distinct requested aspect that has not already been answered; evidence availability alone is not
a reason to add background, benefits, implications, or paraphrases. Stop when the request is
answered, keeping simple definitions, facts, and introductory questions brief.

Every factual sentence must have a directly supporting [E1], [E2], ... reference immediately
before its punctuation. Use only supplied evidence IDs; for multiple passages, write adjacent
references such as [E1][E5]. A citation supports only the sentence in which it appears.
Do not fill gaps from memory or substitute related methods for explicitly named targets.
If the question uses an acronym or named entity without disambiguating context and the candidate
evidence supports multiple identities, state that it is ambiguous and briefly distinguish the
relevant meanings. Do not select one identity merely because its passage has the highest score.
Use a supplied Controller assessment instead of repeating its coverage analysis, while still
checking that each cited passage directly supports your claim. If recovery ran, check only whether
the recovered candidates resolve the listed gap. Mention a gap only when it blocks an important
part of the requested answer; never claim that missing support means the information does not exist.
Start directly with the first supported claim, without an overview, and do not repeat the answer in
a conclusion. Before returning, delete any unsupported factual sentence.

If no supplied evidence supports any requested factual answer, return exactly:
{SAFE_ABSTENTION}

Question: {question}

{context}
"""


def _writer_prompt(state: AgentState) -> str:
    return _render_writer_prompt(state["question"], _writer_context(state))


def _render_answer_citations(answer: str, evidence: list[dict]) -> str:
    return validate_citations(PAGE_CITATION_RE.sub("", answer), evidence)


def _stream_writer(
    prompt: str,
    llm: LLMClient,
    evidence: list[dict],
    emit: Callable[[str], None],
) -> str:
    raw_chunks = []
    pending = ""
    emitted = False
    ends_with_newline = False

    def send(text: str) -> None:
        nonlocal emitted, ends_with_newline
        emit(text)
        emitted = True
        ends_with_newline = text.endswith("\n")

    for chunk in llm.stream(prompt):
        raw_chunks.append(chunk)
        pending += chunk
        while match := STREAM_BOUNDARY_RE.search(pending):
            rendered = _render_answer_citations(pending[:match.start()], evidence)
            if rendered:
                send(rendered + match.group(0))
            pending = pending[match.end():]
    rendered = _render_answer_citations(pending, evidence)
    if rendered:
        send(rendered)
    if emitted and not ends_with_newline:
        send("\n")
    return "".join(raw_chunks).strip()


def writer_node(
    state: AgentState,
    llm: LLMClient,
    emit: Callable[[str], None] | None = None,
) -> dict:
    """Write one grounded draft from the requirement–evidence board."""
    if not state["evidence"]:
        LOGGER.info("[writer] deterministic abstention without evidence")
        return {"answer": SAFE_ABSTENTION}

    prompt = _writer_prompt(state)
    answer = (
        llm.complete(prompt).strip()
        if emit is None
        else _stream_writer(prompt, llm, state["evidence"], emit)
    )
    LOGGER.info("[writer] produced draft")
    return {"answer": answer}


def citation_validator_node(state: AgentState) -> dict:
    """Render known evidence IDs and remove fabricated page citations deterministically."""
    answer = _render_answer_citations(state["answer"], state["evidence"])
    summary = citation_summary(answer, state["evidence"])
    LOGGER.info(
        "[citations] citations=%d sources=%d",
        summary["citations"],
        summary["sources"],
    )
    return {"answer": answer}

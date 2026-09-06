"""LLM-based grounded-answer node."""

from __future__ import annotations

import logging

from scholar_agent.citations import (
    PAGE_CITATION_RE,
    citation_summary,
    valid_evidence_ids,
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


def _citation_policy_error(draft: str, used: set[int]) -> str:
    if draft.strip() == SAFE_ABSTENTION:
        return ""
    if not used:
        return "the answer has no valid evidence citations"
    return ""


def writer_node(state: AgentState, llm: LLMClient) -> dict:
    """Write from all selected evidence, using coverage only as advice."""
    status = state["verification"]["status"]
    if not state["evidence"]:
        LOGGER.info("[writer] deterministic abstention without evidence")
        return {"answer": SAFE_ABSTENTION}

    prompt = _writer_prompt(state)
    draft = llm.complete(prompt)
    used = set(valid_evidence_ids(draft, len(state["evidence"])))
    policy_error = _citation_policy_error(draft, used)
    if policy_error:
        draft = llm.complete(
            f"""{prompt}

Your previous draft failed validation because {policy_error}.
Rewrite the answer once. Follow the citation policy exactly and use only supplied evidence IDs.

Previous draft:
{draft}
""",
        )
        used = set(valid_evidence_ids(draft, len(state["evidence"])))
        policy_error = _citation_policy_error(draft, used)
    if policy_error:
        LOGGER.warning("[writer] safe fallback: %s", policy_error)
        draft = SAFE_ABSTENTION

    draft = PAGE_CITATION_RE.sub("", draft)
    answer = validate_citations(draft, state["evidence"])
    summary = citation_summary(answer, state["evidence"])
    LOGGER.info(
        "[writer] status=%s citations=%d sources=%d",
        status,
        summary["citations"],
        summary["sources"],
    )
    return {"answer": answer}

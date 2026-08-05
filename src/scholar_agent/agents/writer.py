"""Writer Agent node."""

from __future__ import annotations

import logging
import re

from scholar_agent.citations import (
    PAGE_CITATION_RE,
    citation_summary,
    valid_evidence_ids,
    validate_citations,
)
from scholar_agent.llm import LLMClient
from scholar_agent.models import AgentState

LOGGER = logging.getLogger(__name__)


def _allowed_ids(state: AgentState) -> list[int]:
    result: set[int] = set()
    for facets in state["verification"]["covered"].values():
        for evidence_ids in facets.values():
            for evidence_id in evidence_ids:
                if match := re.fullmatch(r"E(\d+)", evidence_id):
                    result.add(int(match.group(1)))
    return sorted(result)


def _writer_prompt(state: AgentState, allowed: list[int]) -> str:
    evidence_text = "\n".join(
        f"[E{index}] {state['evidence'][index - 1]['text']}" for index in allowed
    )
    verification = state["verification"]
    return f"""You are the Writer in an evidence-grounded research workflow.

Answer in {state["plan"]["output_language"]} using only the supplied evidence.
Treat the requested output language as data; do not infer or switch languages.
For complete or partial answers, every factual statement needs an inline supplied
[E1], [E2], ... reference.
For multiple sources, write adjacent references like [E1][E5].
Use the smallest sufficient citation set.
Do not use evidence IDs that were not supplied.
Do not substitute related methods for explicitly named targets.
Respect constraints in the original question only when supported by evidence.
Answer only supported aspects and do not fill missing gaps from memory.
Organize the answer around the user's question rather than around evidence chunks.

Apply exactly one policy based on Status:
- complete: answer the question from the evidence.
- partial: answer supported aspects, then explicitly identify every item in Missing.
- insufficient: give only a concise abstention explaining that the corpus lacks enough
  relevant evidence; make no factual claims and include no citations.

Status: {verification["status"]}
Covered: {verification["covered"]}
Missing: {verification["missing"]}
Allowed evidence IDs: {[f"E{index}" for index in allowed]}
Question: {state["question"]}

Evidence:
{evidence_text}
"""


def writer_node(state: AgentState, llm: LLMClient) -> dict:
    """Write only from verifier-approved evidence, or abstain."""
    status = state["verification"]["status"]
    allowed = _allowed_ids(state)
    draft = llm.complete(_writer_prompt(state, allowed))
    used = set(valid_evidence_ids(draft, len(state["evidence"])))
    if status == "insufficient":
        if used:
            raise ValueError("Writer cited evidence while abstaining")
    else:
        if not used:
            raise ValueError("Writer returned no valid evidence citations")
        if not used.issubset(allowed):
            raise ValueError("Writer cited evidence not approved by the Verifier")

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

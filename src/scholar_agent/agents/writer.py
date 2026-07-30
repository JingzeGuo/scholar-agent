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


def _chinese(state: AgentState) -> bool:
    language = state["plan"]["output_language"]
    return "chinese" in language.casefold() or "中文" in language


def _first_sentence(text: str, limit: int = 260) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"


def _allowed_ids(state: AgentState) -> list[int]:
    result: set[int] = set()
    for facets in state["verification"]["covered"].values():
        for evidence_ids in facets.values():
            for evidence_id in evidence_ids:
                if match := re.fullmatch(r"E(\d+)", evidence_id):
                    result.add(int(match.group(1)))
    return sorted(result)


def _missing_text(state: AgentState) -> str:
    missing = state["verification"]["missing"]
    if not missing:
        return ""
    heading = "缺失证据：" if _chinese(state) else "Missing evidence:"
    return "\n".join([heading, *(f"- {item}" for item in missing)])


def _offline_draft(state: AgentState, allowed: list[int]) -> str:
    status = state["verification"]["status"]
    if status == "insufficient":
        return (
            "当前语料库没有足够相关的证据来回答这个问题。"
            if _chinese(state)
            else "The current corpus does not contain sufficiently relevant evidence "
            "to answer this question."
        )
    lead = "可用证据支持以下内容：" if _chinese(state) else "The available evidence supports:"
    points = [
        f"- {_first_sentence(state['evidence'][index - 1]['text'])} [E{index}]"
        for index in allowed[:4]
    ]
    return "\n".join([lead, *points, _missing_text(state)]).strip()


def _writer_prompt(state: AgentState, allowed: list[int]) -> str:
    evidence_text = "\n".join(
        f"[E{index}] {state['evidence'][index - 1]['text']}"
        for index in allowed
    )
    verification = state["verification"]
    return f"""You are the Writer in an evidence-grounded research workflow.

Answer in {state["plan"]["output_language"]} using only the supplied evidence.
Every factual statement needs an inline supplied [E1], [E2], ... reference.
For multiple sources, write adjacent references like [E1][E5], never [E1, E5].
Do not substitute related methods for these targets: {state["plan"]["targets"]}.
Derive differences only from target-level evidence. Answer only covered aspects;
the system will append the missing-evidence list.

Status: {verification["status"]}
Covered: {verification["covered"]}
Missing: {verification["missing"]}
Question: {state["question"]}

Evidence:
{evidence_text}
"""


def writer_node(state: AgentState, llm: LLMClient | None = None) -> dict:
    """Write only from verifier-approved evidence, or abstain without an LLM."""
    status = state["verification"]["status"]
    allowed = _allowed_ids(state)
    if status == "insufficient" or llm is None:
        draft = _offline_draft(state, allowed)
    else:
        draft = llm.complete(_writer_prompt(state, allowed))
        used = set(valid_evidence_ids(draft, len(state["evidence"])))
        if not used or not used.issubset(allowed):
            LOGGER.warning("[writer] draft used evidence outside verification; using fallback")
            draft = _offline_draft(state, allowed)
        elif status == "partial":
            draft = "\n\n".join([draft.strip(), _missing_text(state)])

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

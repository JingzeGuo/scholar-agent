"""Shared answer generation for retrieval/agent ablations.

The evaluation harness applies this exact prompt *after* each system has
produced its evidence.  That keeps live-LLM answer generation constant while
the retrieval/planning systems vary.  Offline runs intentionally skip this
step and are labelled as heterogeneous extractive/structured-writer runs.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from scholar_agent.llm.client import ChatMessage
from scholar_agent.llm.prompts import build_evidence_prompt_block
from scholar_agent.models.retrieval import RetrievalHit

EVALUATION_ANSWER_PROMPT_ID = "evaluation-grounded-answer-v1"

_SYSTEM_PROMPT = (
    "You are a literature-evaluation answer generator. Answer only from the "
    "retrieved passages supplied by the user. Every factual sentence must use "
    "an inline citation in the exact form [paper_id p.N] or "
    "[paper_id p.N-M]. If the passages do not answer the question, state that "
    "the corpus evidence is insufficient. Never invent papers, pages, or facts."
)


class EvaluationGenerationResult(BaseModel):
    """Secret-free facts about one shared generation call."""

    answer_text: str
    used_llm: bool = False
    generation_model: str | None = None
    prompt_id: str = EVALUATION_ANSWER_PROMPT_ID
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_count_source: str = "estimated"
    fallback_used: bool = False
    skip_reason: str | None = None
    cited_paper_ids: list[str] = Field(default_factory=list)
    citation_count: int = 0
    valid_citation_count: int = 0


def generate_evaluation_answer(
    *,
    question: str,
    hits: list[RetrievalHit],
    llm: Any,
    fallback_answer: str,
    max_tokens: int = 800,
) -> EvaluationGenerationResult:
    """Generate one answer with the shared evaluation prompt.

    ``llm`` deliberately uses a structural interface (``chat`` plus an
    OpenAI-compatible response) so deterministic fake clients can prove that
    the paid path is invoked without making a network call.
    """
    if not hits:
        return EvaluationGenerationResult(
            answer_text=fallback_answer,
            skip_reason="no_retrieved_evidence",
        )

    evidence_rows: list[tuple[str, str, str]] = []
    for hit in hits:
        citation = _inline_citation(hit)
        evidence_rows.append(
            (
                hit.chunk_id,
                hit.paper_id,
                f"Allowed citation: {citation}\n{hit.text}",
            )
        )
    evidence = build_evidence_prompt_block(evidence_rows)
    user_prompt = (
        f"Question:\n{question}\n\n{evidence}\n\n"
        "Write a concise answer. Use only the passages above and their allowed citations."
    )
    response = llm.chat(
        [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ],
        fast=True,
        max_tokens=max_tokens,
    )
    generated = (getattr(response, "content", None) or "").strip()
    fallback_used = not bool(generated)
    answer = generated or fallback_answer
    answer = _ensure_source_references(answer, hits)
    cited_papers, citation_count, valid_citation_count = _citation_facts(answer, hits)

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens <= 0:
        input_tokens = _estimate_tokens(_SYSTEM_PROMPT + "\n" + user_prompt)
        output_tokens = _estimate_tokens(answer)
        total_tokens = input_tokens + output_tokens
        token_source = "estimated"
    else:
        token_source = "provider"

    return EvaluationGenerationResult(
        answer_text=answer,
        used_llm=True,
        generation_model=str(getattr(response, "model", "") or "unknown"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        token_count_source=token_source,
        fallback_used=fallback_used,
        cited_paper_ids=cited_papers,
        citation_count=citation_count,
        valid_citation_count=valid_citation_count,
    )


def requested_generation_model(llm: Any | None) -> str | None:
    """Return the configured fast model without serializing client secrets."""
    config = getattr(llm, "config", None)
    model = getattr(config, "fast_model", None)
    return str(model) if model else None


def _inline_citation(hit: RetrievalHit) -> str:
    pages = (
        f"p.{hit.page_start}"
        if hit.page_start == hit.page_end
        else f"p.{hit.page_start}-{hit.page_end}"
    )
    return f"[{hit.paper_id} {pages}]"


def _ensure_source_references(answer: str, hits: list[RetrievalHit]) -> str:
    if any(_inline_citation(hit) in answer for hit in hits):
        return answer
    refs = "; ".join(_inline_citation(hit) for hit in hits[:5])
    return f"{answer.rstrip()}\n\nSources: {refs}"


_CITATION_RE = re.compile(r"\[(?P<paper>[A-Za-z0-9_.:-]+)\s+p\.(?P<start>\d+)(?:-(?P<end>\d+))?\]")


def _citation_facts(answer: str, hits: list[RetrievalHit]) -> tuple[list[str], int, int]:
    allowed = list(hits)
    papers: list[str] = []
    count = 0
    valid = 0
    for match in _CITATION_RE.finditer(answer):
        count += 1
        paper_id = match.group("paper")
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        papers.append(paper_id)
        if any(
            hit.paper_id == paper_id
            and start >= hit.page_start
            and end <= hit.page_end
            and end >= start
            for hit in allowed
        ):
            valid += 1
    return list(dict.fromkeys(papers)), count, valid


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0

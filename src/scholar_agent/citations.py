"""Deterministic evidence-ID validation and page-aware citation rendering."""

from __future__ import annotations

import re

EVIDENCE_CITATION_RE = re.compile(r"\[E(\d+)\]")
PAGE_CITATION_RE = re.compile(r"\[([^\[\]]+\.pdf) p\.(\d+)\]")


def validate_citations(answer: str, evidence: list[dict]) -> str:
    """Drop unknown evidence IDs and render known IDs from real chunk metadata."""
    citations = {
        index: f"[{item['paper']} p.{item['page']}]"
        for index, item in enumerate(evidence, start=1)
        if item.get("paper") and isinstance(item.get("page"), int)
    }

    def replace(match: re.Match[str]) -> str:
        return citations.get(int(match.group(1)), "")

    answer = EVIDENCE_CITATION_RE.sub(replace, answer)
    valid_pages = {
        (str(item["paper"]), int(item["page"]))
        for item in evidence
        if item.get("paper") and isinstance(item.get("page"), int)
    }

    def keep_real_page(match: re.Match[str]) -> str:
        provenance = (match.group(1), int(match.group(2)))
        return match.group(0) if provenance in valid_pages else ""

    answer = PAGE_CITATION_RE.sub(keep_real_page, answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r" +([,.;:])", r"\1", answer)
    return answer.strip()


def valid_evidence_ids(answer: str, evidence_count: int) -> list[int]:
    """Return unique in-range evidence references in their first-seen order."""
    result: list[int] = []
    for value in EVIDENCE_CITATION_RE.findall(answer):
        evidence_id = int(value)
        if 1 <= evidence_id <= evidence_count and evidence_id not in result:
            result.append(evidence_id)
    return result


def cited_pages(answer: str) -> list[tuple[str, int]]:
    return [(paper, int(page)) for paper, page in PAGE_CITATION_RE.findall(answer)]


def citation_summary(answer: str, evidence: list[dict]) -> dict[str, int | bool]:
    """Summarize validated provenance without creating a citation data model."""
    pages = cited_pages(answer)
    valid_pages = {
        (str(item["paper"]), int(item["page"]))
        for item in evidence
        if item.get("paper") and isinstance(item.get("page"), int)
    }
    return {
        "citations": len(pages),
        "sources": len({paper for paper, _ in pages}),
        "all_grounded": all(page in valid_pages for page in pages),
    }

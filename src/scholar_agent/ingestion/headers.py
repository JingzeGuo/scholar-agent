"""Header / footer normalization across pages."""

from __future__ import annotations

import re
from collections import Counter

from scholar_agent.models.corpus import PaperPage

_LINE_SPLIT = re.compile(r"\n+")


def _edge_lines(text: str, *, n: int = 2) -> tuple[list[str], list[str]]:
    lines = [ln.strip() for ln in _LINE_SPLIT.split(text) if ln.strip()]
    if not lines:
        return [], []
    head = lines[:n]
    tail = lines[-n:] if len(lines) > n else []
    return head, tail


def detect_repeated_edges(
    pages: list[PaperPage],
    *,
    min_fraction: float = 0.5,
    max_line_len: int = 120,
) -> set[str]:
    """Return line strings that look like repeated headers/footers."""
    if len(pages) < 3:
        return set()
    counter: Counter[str] = Counter()
    eligible = 0
    for page in pages:
        if page.is_empty:
            continue
        eligible += 1
        head, tail = _edge_lines(page.text)
        for line in head + tail:
            if 3 <= len(line) <= max_line_len:
                # Normalize page numbers in repeated lines
                norm = re.sub(r"\b\d+\b", "#", line).strip().lower()
                counter[norm] += 1
    if eligible == 0:
        return set()
    threshold = max(2, int(eligible * min_fraction))
    return {line for line, count in counter.items() if count >= threshold}


def strip_headers_footers(
    pages: list[PaperPage], repeated: set[str] | None = None
) -> list[PaperPage]:
    """Return pages with repeated header/footer lines removed."""
    if repeated is None:
        repeated = detect_repeated_edges(pages)
    if not repeated:
        return pages

    cleaned: list[PaperPage] = []
    for page in pages:
        if page.is_empty:
            cleaned.append(page)
            continue
        kept: list[str] = []
        for line in _LINE_SPLIT.split(page.text):
            stripped = line.strip()
            if not stripped:
                if kept and kept[-1] != "":
                    kept.append("")
                continue
            norm = re.sub(r"\b\d+\b", "#", stripped).strip().lower()
            if norm in repeated:
                continue
            kept.append(stripped)
        text = "\n".join(kept).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        cleaned.append(
            page.model_copy(
                update={
                    "text": text,
                    "char_count": len(text),
                    "is_empty": not text.strip(),
                }
            )
        )
    return cleaned

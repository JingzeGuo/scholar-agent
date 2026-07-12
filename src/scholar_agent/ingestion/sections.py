"""Heuristic section heading detection and page→section blocking."""

from __future__ import annotations

import re

from scholar_agent.ingestion.tokens import count_tokens
from scholar_agent.models.corpus import PaperPage
from scholar_agent.models.ingestion import SectionBlock

# Numbered academic headings: "1 Introduction", "1. Introduction", "1.2 Method"
_NUMBERED = re.compile(
    r"^(?:"
    r"\d{1,2}(?:\.\d{1,2}){0,3}\.?"  # 1 / 1. / 1.2 / 1.2.3
    r"|[IVXLC]{1,6}\."  # IV.
    r")\s+[A-Z][A-Za-z][\w\s\-:,]{1,80}$"
)
_NUMBERED_LOOSE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?\s+[A-Z][\w\s\-:,]{2,80}$")
_NAMED = re.compile(
    r"^(?:"
    r"Abstract|Introduction|Related Work|Background|Preliminaries|"
    r"Method(?:s|ology)?|Approach|Model|Experiments?|Evaluation|"
    r"Results?|Discussion|Conclusion|Conclusions|Limitations|"
    r"Future Work|References|Bibliography|Acknowledgments?|"
    r"Appendix(?:\s+[A-Z0-9]+)?|Supplementary(?:\s+Material)?"
    r")\s*$",
    re.I,
)
_ALL_CAPS_NAMED = re.compile(
    r"^(?:"
    r"ABSTRACT|INTRODUCTION|RELATED WORK|BACKGROUND|METHOD|METHODS|"
    r"METHODOLOGY|EXPERIMENTS?|EVALUATION|RESULTS?|DISCUSSION|"
    r"CONCLUSION|CONCLUSIONS|REFERENCES|APPENDIX"
    r")\s*$"
)


def is_section_heading(line: str) -> bool:
    """Return True only for high-precision academic headings.

    Intentionally conservative: false negatives (missing a heading) are
    preferable to false positives that shatter tables and formulas into chunks.
    """
    text = line.strip()
    if not text or len(text) > 90:
        return False
    # Reject lines that look like prose, emails, equations, tables
    if "@" in text or "=" in text or text.count(" ") > 12:
        return False
    if (
        re.search(r"\d+\.\d+", text)
        and not _NUMBERED_LOOSE.match(text)
        and not re.match(r"^\d+(\.\d+)*\s+[A-Za-z]", text)
    ):
        # decimal numbers often table values unless numbered heading
        return False
    if _NAMED.match(text) or _ALL_CAPS_NAMED.match(text):
        return True
    return bool(_NUMBERED.match(text) or _NUMBERED_LOOSE.match(text))


def pages_to_sections(pages: list[PaperPage]) -> list[SectionBlock]:
    """Split cleaned pages into section blocks with page provenance."""
    blocks: list[SectionBlock] = []
    current_title: str | None = None
    current_lines: list[str] = []
    page_start: int | None = None
    page_end: int | None = None

    def flush() -> None:
        nonlocal current_lines, page_start, page_end, current_title
        text = "\n".join(current_lines).strip()
        if text and page_start is not None and page_end is not None:
            blocks.append(
                SectionBlock(
                    title=current_title,
                    page_start=page_start,
                    page_end=page_end,
                    text=text,
                )
            )
        current_lines = []
        page_start = None
        page_end = None

    for page in pages:
        if page.is_empty:
            continue
        for line in page.text.split("\n"):
            stripped = line.strip()
            if not stripped:
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
                continue
            if is_section_heading(stripped):
                flush()
                current_title = stripped
                page_start = page.page_number
                page_end = page.page_number
                continue
            if page_start is None:
                page_start = page.page_number
            page_end = page.page_number
            current_lines.append(stripped)
    flush()
    return merge_small_sections(blocks)


def merge_small_sections(
    sections: list[SectionBlock],
    *,
    min_tokens: int = 80,
) -> list[SectionBlock]:
    """Merge undersized sections into the previous block to avoid micro-chunks."""
    if not sections:
        return []
    merged: list[SectionBlock] = []
    for section in sections:
        tokens = count_tokens(section.text)
        if merged and tokens < min_tokens:
            prev = merged[-1]
            merged[-1] = SectionBlock(
                title=prev.title,
                page_start=prev.page_start,
                page_end=max(prev.page_end, section.page_end),
                text=(prev.text + "\n\n" + section.text).strip(),
            )
        else:
            merged.append(section)
    return merged

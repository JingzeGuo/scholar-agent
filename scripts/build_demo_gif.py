#!/usr/bin/env python3
"""Build a small replay-derived GIF for the ScholarAgent README.

This is deliberately not a screenshot or a simulation of a live Streamlit run.
Every query, plan step, evidence claim, corrective event, and citation shown in
the animation is read from the committed ``selfrag_vs_crag`` replay JSON.
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scholar_agent.app.demo_models import SavedDemoRun
from scholar_agent.app.demo_runs import load_saved_run
from scholar_agent.models.base import EventType

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "data" / "demo" / "runs" / "selfrag_vs_crag.json"
DEFAULT_OUTPUT = REPO / "docs" / "assets" / "scholaragent_replay.gif"

BACKGROUND = "#07141e"
HEADER = "#0d2735"
PANEL = "#102f40"
PANEL_ALT = "#143a4b"
INK = "#eef7f8"
MUTED = "#9fc0c9"
ACCENT = "#43d6a5"
WARNING = "#ffbe5c"
RAIL_IDLE = "#466472"


@dataclass(frozen=True)
class ReplayStory:
    title: str
    demo_id: str
    query: str
    sub_questions: tuple[str, ...]
    evidence_lines: tuple[str, ...]
    evidence_provenance: tuple[str, ...]
    verification_summary: str
    corrective_summary: str
    corrective_queries: tuple[str, ...]
    cited_claims: tuple[str, ...]
    corpus_fingerprint: str


def _page_label(start: int, end: int) -> str:
    return f"p.{start}" if start == end else f"pp.{start}-{end}"


def extract_story(run: SavedDemoRun) -> ReplayStory:
    """Validate and project the structured replay into animation-safe text."""
    session = run.session
    if not run.offline or not session.offline_replay:
        raise ValueError("demo GIF input must be an offline replay")
    if session.plan is None or not session.plan.sub_questions:
        raise ValueError("demo GIF input has no planner output")
    if len(session.evidence) < 2:
        raise ValueError("demo GIF input needs both comparison evidence items")
    if session.final_answer is None or not session.final_answer.claims:
        raise ValueError("demo GIF input has no cited final answer")

    verification_events = [
        event
        for event in session.events
        if event.event_type == EventType.VERIFICATION
        and event.payload.get("is_sufficient") is False
    ]
    corrective_events = [
        event for event in session.events if event.event_type == EventType.CORRECTIVE
    ]
    if not verification_events or not corrective_events:
        raise ValueError("demo GIF input must contain a verifier-triggered corrective pass")

    cards_by_evidence = {card.evidence_id: card for card in session.source_cards}
    evidence_lines: list[str] = []
    provenance: list[str] = []
    for item in session.evidence:
        evidence_lines.append(item.claim)
        provenance.append(
            f"{item.paper_id} | {item.chunk_id} | {_page_label(item.page_start, item.page_end)}"
        )

    cited_claims: list[str] = []
    for claim in session.final_answer.claims:
        references = []
        for evidence_id in claim.evidence_ids:
            card = cards_by_evidence.get(evidence_id)
            if card is None:
                raise ValueError(f"final claim references missing source card: {evidence_id}")
            references.append(f"{card.paper_id} {_page_label(card.page_start, card.page_end)}")
        cited_claims.append(f"{claim.text} [{'; '.join(references)}]")

    corrective_queries = tuple(session.trace.corrective_queries)
    if not corrective_queries:
        raise ValueError("demo GIF input has no structured corrective query")

    return ReplayStory(
        title=run.title,
        demo_id=run.demo_id,
        query=run.query,
        sub_questions=tuple(item.question for item in session.plan.sub_questions),
        evidence_lines=tuple(evidence_lines),
        evidence_provenance=tuple(provenance),
        verification_summary=verification_events[0].summary,
        corrective_summary=corrective_events[0].summary,
        corrective_queries=corrective_queries,
        cited_claims=tuple(cited_claims),
        corpus_fingerprint=run.corpus_fingerprint or "unverified",
    )


def _fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    return (
        ImageFont.load_default(size=29),
        ImageFont.load_default(size=20),
        ImageFont.load_default(size=16),
    )


def _wrapped(value: str, *, width: int) -> list[str]:
    return textwrap.wrap(
        " ".join(value.split()),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]


def _text_block(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: str,
    *,
    font: ImageFont.ImageFont,
    fill: str = INK,
    width: int = 78,
    spacing: int = 5,
) -> int:
    x, y = position
    lines = _wrapped(value, width=width)
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + spacing
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _base_frame(story: ReplayStory, active: int, *, width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _fonts()

    draw.rounded_rectangle((18, 16, width - 18, 88), radius=14, fill=HEADER)
    draw.text((38, 29), "ScholarAgent", font=title_font, fill=INK)
    badge = "OFFLINE REPLAY | structured JSON"
    badge_width = draw.textlength(badge, font=small_font) + 28
    draw.rounded_rectangle(
        (width - badge_width - 38, 32, width - 38, 69),
        radius=12,
        outline=ACCENT,
        width=2,
    )
    draw.text((width - badge_width - 24, 40), badge, font=small_font, fill=ACCENT)

    stages = ("Query", "Planner", "Research", "Verify + fix", "Research", "Cited answer")
    rail_y = 119
    left = 47
    right = width - 47
    step = (right - left) / (len(stages) - 1)
    draw.line((left, rail_y, right, rail_y), fill=RAIL_IDLE, width=3)
    for index, label in enumerate(stages):
        x = int(left + step * index)
        reached = index <= active
        draw.ellipse(
            (x - 8, rail_y - 8, x + 8, rail_y + 8),
            fill=ACCENT if reached else BACKGROUND,
            outline=ACCENT if reached else RAIL_IDLE,
            width=2,
        )
        label_width = draw.textlength(label, font=small_font)
        draw.text(
            (x - label_width / 2, rail_y + 15),
            label,
            font=small_font,
            fill=INK if reached else MUTED,
        )

    draw.rounded_rectangle((25, 166, width - 25, height - 42), radius=16, fill=PANEL)
    footer = f"Replay-derived asset | {story.demo_id} | corpus {story.corpus_fingerprint[:12]}"
    draw.text((28, height - 29), footer, font=small_font, fill=MUTED)
    return image


def _draw_heading(draw: ImageDraw.ImageDraw, heading: str, detail: str) -> int:
    title_font, body_font, small_font = _fonts()
    draw.text((52, 187), heading, font=title_font, fill=ACCENT)
    draw.text((52, 226), detail, font=small_font, fill=MUTED)
    return 263


def render_frames(story: ReplayStory, *, width: int = 960, height: int = 540) -> list[Image.Image]:
    """Render the six truthful stages of the replay story."""
    if width < 720 or height < 400:
        raise ValueError("demo GIF must be at least 720x400")
    title_font, body_font, small_font = _fonts()
    frames: list[Image.Image] = []

    frame = _base_frame(story, 0, width=width, height=height)
    draw = ImageDraw.Draw(frame)
    y = _draw_heading(draw, "Replay query", story.title)
    draw.rounded_rectangle((48, y, width - 48, y + 118), radius=12, fill=PANEL_ALT)
    _text_block(draw, (70, y + 28), story.query, font=body_font, width=70)
    frames.append(frame)

    frame = _base_frame(story, 1, width=width, height=height)
    draw = ImageDraw.Draw(frame)
    y = _draw_heading(draw, "Planner", f"{len(story.sub_questions)} comparison sub-questions")
    for index, question in enumerate(story.sub_questions[:3], start=1):
        draw.rounded_rectangle((48, y, width - 48, y + 74), radius=10, fill=PANEL_ALT)
        draw.text((67, y + 13), f"SQ{index}", font=small_font, fill=ACCENT)
        _text_block(draw, (122, y + 12), question, font=body_font, width=65)
        y += 88
    frames.append(frame)

    frame = _base_frame(story, 2, width=width, height=height)
    draw = ImageDraw.Draw(frame)
    y = _draw_heading(draw, "Researcher", "First structured evidence item")
    draw.rounded_rectangle((48, y, width - 48, y + 150), radius=12, fill=PANEL_ALT)
    y = _text_block(
        draw,
        (69, y + 18),
        story.evidence_lines[0],
        font=body_font,
        width=70,
    )
    _text_block(
        draw,
        (69, y + 12),
        story.evidence_provenance[0],
        font=small_font,
        fill=MUTED,
        width=90,
    )
    frames.append(frame)

    frame = _base_frame(story, 3, width=width, height=height)
    draw = ImageDraw.Draw(frame)
    y = _draw_heading(draw, "Verifier -> corrective", story.verification_summary)
    draw.rounded_rectangle((48, y, width - 48, y + 79), radius=12, fill="#483928")
    _text_block(
        draw,
        (69, y + 18),
        story.corrective_summary,
        font=body_font,
        fill=WARNING,
        width=70,
    )
    y += 95
    draw.text((55, y), "Corrective query from replay:", font=small_font, fill=MUTED)
    _text_block(
        draw,
        (69, y + 28),
        story.corrective_queries[0],
        font=body_font,
        width=70,
    )
    frames.append(frame)

    frame = _base_frame(story, 4, width=width, height=height)
    draw = ImageDraw.Draw(frame)
    y = _draw_heading(draw, "Researcher", "Corrective evidence closes the comparison gap")
    draw.rounded_rectangle((48, y, width - 48, y + 150), radius=12, fill=PANEL_ALT)
    y = _text_block(
        draw,
        (69, y + 18),
        story.evidence_lines[1],
        font=body_font,
        width=70,
    )
    _text_block(
        draw,
        (69, y + 12),
        story.evidence_provenance[1],
        font=small_font,
        fill=MUTED,
        width=90,
    )
    frames.append(frame)

    frame = _base_frame(story, 5, width=width, height=height)
    draw = ImageDraw.Draw(frame)
    y = _draw_heading(draw, "Cited answer", f"{len(story.cited_claims)} claims with source cards")
    for claim in story.cited_claims[:3]:
        draw.rounded_rectangle((48, y, width - 48, y + 87), radius=10, fill=PANEL_ALT)
        _text_block(draw, (69, y + 13), claim, font=small_font, width=91)
        y += 99
    draw.text(
        (width - 198, height - 30),
        "VERIFIED REPLAY",
        font=small_font,
        fill=ACCENT,
    )
    frames.append(frame)
    return frames


def build_demo_gif(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    width: int = 960,
    height: int = 540,
) -> int:
    """Build the replay GIF and return its frame count."""
    story = extract_story(load_saved_run(input_path))
    frames = render_frames(story, width=width, height=height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=[1100, 1300, 1400, 1500, 1400, 2200],
        loop=0,
        optimize=True,
        disposal=2,
    )
    return len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    frame_count = build_demo_gif(
        args.input,
        args.out,
        width=args.width,
        height=args.height,
    )
    print(f"wrote {args.out} ({frame_count} replay-derived frames)")


if __name__ == "__main__":
    main()

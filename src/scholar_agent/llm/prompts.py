"""Shared prompt fragments.

Keep prompts free of untrusted retrieved text until call sites delimit it.
Chain-of-thought instructions must never request private reasoning in user-facing
outputs; agents should emit structured decisions only.
"""

from __future__ import annotations

SYSTEM_STRUCTURED_JSON = (
    "You are a careful research assistant. Respond with valid JSON only. "
    "Do not include markdown fences or private chain-of-thought."
)

SYSTEM_TOOL_USE = (
    "You are a research agent with tools. Choose tools when needed, "
    "ground claims in tool results, and stop when evidence is sufficient. "
    "Never invent citations or paper titles."
)

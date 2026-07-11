"""Structured JSON parsing and light repair for LLM outputs."""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


class StructuredOutputError(ValueError):
    """Raised when model output cannot be parsed into the target schema."""


def extract_json_text(raw: str) -> str:
    """Extract a JSON object/array from raw model text.

    Handles fenced markdown blocks and leading/trailing prose.
    """
    text = raw.strip()
    if not text:
        raise StructuredOutputError("empty model output")

    fence = _JSON_FENCE_RE.search(text)
    if fence:
        return fence.group(1).strip()

    # Find first JSON object or array
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]

    return text


def parse_structured_json(raw: str, model_type: type[T]) -> T:
    """Parse raw model text into a Pydantic model, with light repair."""
    candidates = [raw]
    with contextlib.suppress(StructuredOutputError):
        candidates.insert(0, extract_json_text(raw))

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            # Common repair: trailing commas
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError as exc2:
                last_error = exc2
                continue
        try:
            return model_type.model_validate(data)
        except ValidationError as exc:
            last_error = exc
            continue

    raise StructuredOutputError(
        f"Failed to parse structured output for {model_type.__name__}: {last_error}"
    )


def request_structured_json_with_retry(
    request: Callable[[], str],
    model_type: type[T],
    *,
    max_attempts: int = 2,
) -> T:
    """Request and parse structured output, retrying malformed model responses."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    last_error: StructuredOutputError | None = None
    for _ in range(max_attempts):
        try:
            return parse_structured_json(request(), model_type)
        except StructuredOutputError as exc:
            last_error = exc
    raise StructuredOutputError(
        f"Structured output remained invalid after {max_attempts} attempts: {last_error}"
    )

"""Structured JSON parsing and safe failure classification for LLM outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_SAFE_PATH_PART_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_ERROR_PATHS = 8
_MAX_PATH_LENGTH = 160
_JSON_DECODE_FAILURE = object()


class StructuredOutputErrorCode(StrEnum):
    """Stable, secret-free structured-output failure reasons."""

    JSON_DECODE_FAILED = "json_decode_failed"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    UNKNOWN_ENTITY_ID = "unknown_entity_id"
    UNKNOWN_REQUIREMENT_KEY = "unknown_requirement_key"
    UNKNOWN_EVIDENCE_ID = "unknown_evidence_id"


class StructuredOutputError(ValueError):
    """A classified error that never retains raw model output or field values."""

    def __init__(
        self,
        code: StructuredOutputErrorCode,
        *,
        field_paths: tuple[str, ...] | list[str] = (),
    ) -> None:
        self.code = code
        self.field_paths = _safe_field_paths(field_paths)
        suffix = (
            f" fields={','.join(self.field_paths)}" if self.field_paths else ""
        )
        super().__init__(f"{code.value}{suffix}")


def extract_json_text(raw: str) -> str:
    """Extract a JSON object/array from raw model text.

    Handles fenced markdown blocks and leading/trailing prose.
    """
    text = raw.strip()
    if not text:
        raise StructuredOutputError(StructuredOutputErrorCode.JSON_DECODE_FAILED)

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
    """Parse model text without retaining its content in raised exceptions."""
    candidate = extract_json_text(raw)
    data = _decode_json(candidate)
    if data is _JSON_DECODE_FAILURE:
        raw = ""
        candidate = ""
        raise StructuredOutputError(StructuredOutputErrorCode.JSON_DECODE_FAILED)

    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        missing_errors = [
            error for error in errors if str(error.get("type", "")) == "missing"
        ]
        selected_errors = missing_errors or errors
        code = (
            StructuredOutputErrorCode.MISSING_REQUIRED_FIELD
            if missing_errors
            else StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED
        )
        paths = [
            _format_validation_location(error.get("loc", ()))
            for error in selected_errors
        ]
    # Raise after leaving the ValidationError handler so the safe exception
    # does not retain the provider-authored values through ``__context__``.
    raw = ""
    candidate = ""
    data = None
    raise StructuredOutputError(code, field_paths=paths)


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
    if last_error is None:  # pragma: no cover - max_attempts is validated above
        raise AssertionError("structured output retry ended without an attempt")
    raise last_error


def _format_validation_location(location: object) -> str:
    if not isinstance(location, (tuple, list)):
        return "$"
    path = ""
    for part in location:
        if isinstance(part, int):
            path += f"[{part}]"
            continue
        cleaned = _SAFE_PATH_PART_RE.sub("_", str(part)).strip("_")
        if not cleaned:
            cleaned = "field"
        path += f".{cleaned}" if path else cleaned
    return path[:_MAX_PATH_LENGTH] or "$"


def _decode_json(candidate: str) -> object:
    # The only intentional repair is a trailing comma before ``}``/``]``.
    # Decoder exceptions stay inside this helper and are never chained.
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    for value in (candidate, repaired):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            continue
    return _JSON_DECODE_FAILURE


def _safe_field_paths(paths: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    safe: list[str] = []
    for path in paths:
        cleaned = re.sub(r"[^A-Za-z0-9_.\[\]-]+", "_", str(path))[
            :_MAX_PATH_LENGTH
        ]
        if cleaned and cleaned not in safe:
            safe.append(cleaned)
        if len(safe) >= _MAX_ERROR_PATHS:
            break
    return tuple(safe)

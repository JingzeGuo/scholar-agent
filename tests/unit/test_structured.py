"""Tests for structured JSON parsing helpers."""

from __future__ import annotations

import pytest

from scholar_agent.llm.structured import (
    StructuredOutputError,
    parse_structured_json,
    request_structured_json_with_retry,
)
from scholar_agent.models import CompatibilityDecision


def test_parse_plain_json() -> None:
    raw = '{"action": "retrieve", "reason": "need evidence", "need_more_evidence": true}'
    result = parse_structured_json(raw, CompatibilityDecision)
    assert result.action == "retrieve"
    assert result.need_more_evidence is True


def test_parse_fenced_json() -> None:
    raw = """Here is the decision:
```json
{"action": "verify", "reason": "enough evidence", "need_more_evidence": false}
```
"""
    result = parse_structured_json(raw, CompatibilityDecision)
    assert result.action == "verify"


def test_parse_trailing_comma_repair() -> None:
    raw = '{"action": "finish", "reason": "done", "need_more_evidence": false,}'
    result = parse_structured_json(raw, CompatibilityDecision)
    assert result.action == "finish"


def test_invalid_json_raises() -> None:
    with pytest.raises(StructuredOutputError):
        parse_structured_json("not json at all", CompatibilityDecision)


def test_schema_mismatch_raises() -> None:
    with pytest.raises(StructuredOutputError):
        parse_structured_json('{"action": "explode", "reason": "x"}', CompatibilityDecision)


def test_request_structured_json_retries_malformed_response() -> None:
    responses = iter(
        [
            "not json",
            '{"action":"finish","reason":"done","need_more_evidence":false}',
        ]
    )

    result = request_structured_json_with_retry(
        lambda: next(responses),
        CompatibilityDecision,
        max_attempts=2,
    )

    assert result.action == "finish"


def test_request_structured_json_exhausts_attempts() -> None:
    with pytest.raises(StructuredOutputError, match="after 2 attempts"):
        request_structured_json_with_retry(
            lambda: "still invalid",
            CompatibilityDecision,
            max_attempts=2,
        )

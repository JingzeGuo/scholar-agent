"""Tests for structured JSON parsing helpers."""

from __future__ import annotations

import pytest

from scholar_agent.llm.structured import (
    StructuredOutputError,
    StructuredOutputErrorCode,
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
    raw = "not json private-provider-value"
    with pytest.raises(StructuredOutputError) as exc_info:
        parse_structured_json(raw, CompatibilityDecision)
    assert exc_info.value.code == StructuredOutputErrorCode.JSON_DECODE_FAILED
    assert exc_info.value.field_paths == ()
    assert raw not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_schema_mismatch_raises() -> None:
    private_value = "private-invalid-action"
    with pytest.raises(StructuredOutputError) as exc_info:
        parse_structured_json(
            f'{{"action": "{private_value}", "reason": "x"}}',
            CompatibilityDecision,
        )
    assert exc_info.value.code == StructuredOutputErrorCode.SCHEMA_VALIDATION_FAILED
    assert exc_info.value.field_paths == ("action",)
    assert private_value not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_missing_required_field_reports_only_field_path() -> None:
    private_value = "private-reason-value"
    with pytest.raises(StructuredOutputError) as exc_info:
        parse_structured_json(
            f'{{"action": "finish", "extra": "{private_value}"}}',
            CompatibilityDecision,
        )
    assert exc_info.value.code == StructuredOutputErrorCode.MISSING_REQUIRED_FIELD
    assert exc_info.value.field_paths == ("reason",)
    assert str(exc_info.value) == "missing_required_field fields=reason"
    assert private_value not in str(exc_info.value)


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
    with pytest.raises(StructuredOutputError) as exc_info:
        request_structured_json_with_retry(
            lambda: "still invalid",
            CompatibilityDecision,
            max_attempts=2,
        )
    assert exc_info.value.code == StructuredOutputErrorCode.JSON_DECODE_FAILED

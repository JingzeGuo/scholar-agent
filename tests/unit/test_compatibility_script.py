"""Regression tests for strict compatibility acceptance semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.deepseek_compatibility import (
    CompatibilityReport,
    check_multi_turn_tools,
    check_tool_calling,
)

from scholar_agent.llm.client import ChatResponse


class FakeClient:
    def __init__(self, responses: list[ChatResponse]) -> None:
        self.responses = iter(responses)

    def chat(self, *_args: Any, **_kwargs: Any) -> ChatResponse:
        return next(self.responses)


def _response(*, tool_calls: list[dict[str, Any]] | None = None) -> ChatResponse:
    return ChatResponse(
        content="answer",
        model="fake",
        tool_calls=tool_calls or [],
    )


def test_tool_calling_fails_when_provider_does_not_call_tool() -> None:
    result = check_tool_calling(FakeClient([_response()]), "fake")  # type: ignore[arg-type]
    assert result.passed is False
    assert "no tool call" in result.detail


def test_multi_turn_fails_when_tool_path_is_not_exercised() -> None:
    result = check_multi_turn_tools(FakeClient([_response()]), "fake")  # type: ignore[arg-type]
    assert result.passed is False
    assert "no tool call" in result.detail


def test_compatibility_report_is_machine_readable(tmp_path: Path) -> None:
    report = CompatibilityReport()
    path = tmp_path / "report.json"
    report.write_json(path, model="fake")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model"] == "fake"
    assert payload["passed"] is True
    assert payload["results"] == []

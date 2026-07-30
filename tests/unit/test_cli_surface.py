"""Tests for the interview-focused public CLI surface."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from scholar_agent.cli import app
from scholar_agent.config import AppConfig

runner = CliRunner()


def test_public_help_keeps_interview_commands_and_removes_stage_artifacts() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("retrieve", "ask-naive", "ask", "evaluate", "demo"):
        assert f"│ {command}" in result.output
    for command in ("prototype", "ablate", "research"):
        assert f"│ {command}" not in result.output


def test_removed_top_level_commands_are_not_callable() -> None:
    for command in ("prototype", "ablate", "research"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_removed_nested_commands_are_not_callable() -> None:
    for group, command in (("corpus", "summary"), ("graph", "stats")):
        result = runner.invoke(app, [group, command, "--help"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    corpus_help = runner.invoke(app, ["corpus", "--help"])
    graph_help = runner.invoke(app, ["graph", "--help"])
    assert "│ summary" not in corpus_help.output
    assert "│ stats" not in graph_help.output


def _patch_ask_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_key: str | None,
    run_error: Exception | None = None,
    run_result: object | None = None,
) -> dict[str, Any]:
    """Replace index/workflow I/O while retaining real CLI runtime selection."""
    config = AppConfig()
    config = config.model_copy(
        update={"llm": config.llm.model_copy(update={"api_key": api_key})}
    )
    captured: dict[str, Any] = {}

    class FakeResult:
        def model_dump(self, *, mode: str) -> dict[str, Any]:
            assert mode == "json"
            return {"run_id": "run_cli_runtime"}

    class FakeWorkflow:
        def __init__(
            self,
            toolkit: object,
            *,
            config: object,
            planner: object,
            writer: object,
        ) -> None:
            captured.update(
                {
                    "toolkit": toolkit,
                    "config": config,
                    "planner": planner,
                    "writer": writer,
                }
            )

        def run(self, query: str) -> object:
            captured["query"] = query
            if run_error is not None:
                raise run_error
            return run_result or FakeResult()

    monkeypatch.setattr("scholar_agent.cli.load_config", lambda _path: config)
    monkeypatch.setattr("scholar_agent.cli.setup_logging", lambda _config: None)
    monkeypatch.setattr(
        "scholar_agent.retrieval.index_builder.load_toolkit",
        lambda **_kwargs: "fake-toolkit",
    )
    monkeypatch.setattr("scholar_agent.agents.workflow.ResearchWorkflow", FakeWorkflow)
    return captured


@pytest.mark.parametrize(
    ("api_key", "extra_args"),
    [
        (None, []),
        ("configured-key", ["--agent-mode", "deterministic"]),
        ("configured-key", ["--offline"]),
    ],
)
def test_ask_deterministic_modes_never_create_llm_client(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str | None,
    extra_args: list[str],
) -> None:
    captured = _patch_ask_runtime(monkeypatch, api_key=api_key)

    def unexpected_client(_config: object) -> object:
        raise AssertionError("deterministic mode must not construct an LLM client")

    monkeypatch.setattr(
        "scholar_agent.llm.client.create_llm_client",
        unexpected_client,
    )
    result = runner.invoke(app, ["ask", "What is Self-RAG?", "--json", *extra_args])

    assert result.exit_code == 0, result.output
    assert captured["planner"].llm is None
    assert captured["writer"].llm is None
    assert captured["planner"].strict_llm is False
    assert captured["writer"].strict_llm is False


def test_ask_auto_shares_one_llm_client_between_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_ask_runtime(monkeypatch, api_key="configured-key")
    client = object()
    calls: list[object] = []

    def fake_create(config: object) -> object:
        calls.append(config)
        return client

    monkeypatch.setattr("scholar_agent.llm.client.create_llm_client", fake_create)
    result = runner.invoke(app, ["ask", "Compare Self-RAG versus CRAG", "--json"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert captured["planner"].llm is client
    assert captured["writer"].llm is client
    assert captured["planner"].strict_llm is False
    assert captured["writer"].strict_llm is False


def test_ask_llm_mode_requires_key_before_loading_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ask_runtime(monkeypatch, api_key=None)

    def unexpected_load(**_kwargs: object) -> object:
        raise AssertionError("indexes should not load when strict LLM config is invalid")

    monkeypatch.setattr(
        "scholar_agent.retrieval.index_builder.load_toolkit",
        unexpected_load,
    )
    result = runner.invoke(
        app,
        ["ask", "What is Self-RAG?", "--agent-mode", "llm", "--json"],
    )

    assert result.exit_code == 2
    assert "requires an LLM API key" in result.output


def test_ask_llm_mode_reports_sanitized_component_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scholar_agent.agents.writer import WriterLLMError

    captured = _patch_ask_runtime(
        monkeypatch,
        api_key="configured-key",
        run_error=WriterLLMError("raw-provider-response-secret"),
    )
    client = object()
    monkeypatch.setattr(
        "scholar_agent.llm.client.create_llm_client",
        lambda _config: client,
    )
    result = runner.invoke(
        app,
        ["ask", "What is Self-RAG?", "--agent-mode", "llm", "--json"],
    )

    assert result.exit_code == 1
    assert "LLM agent execution failed in strict mode (WriterLLMError)" in result.output
    assert "raw-provider-response-secret" not in result.output
    assert captured["planner"].strict_llm is True
    assert captured["writer"].strict_llm is True


def test_ask_llm_mode_does_not_misreport_unrelated_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ask_runtime(
        monkeypatch,
        api_key="configured-key",
        run_error=RuntimeError("unrelated index corruption"),
    )
    monkeypatch.setattr(
        "scholar_agent.llm.client.create_llm_client",
        lambda _config: object(),
    )
    result = runner.invoke(
        app,
        ["ask", "What is Self-RAG?", "--agent-mode", "llm", "--json"],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert "LLM agent execution failed" not in result.output


def test_ask_text_output_reports_actual_component_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scholar_agent.models.answer import AnswerStatus
    from scholar_agent.models.base import EventType, ExecutionEvent, TokenUsage

    events = [
        ExecutionEvent(
            run_id="run_cli_runtime",
            event_type=EventType.PLAN_CREATED,
            component="planner",
            summary="planner degraded",
            payload={
                "backend": "deterministic",
                "model": "fast-model",
                "fallback_reason": "invalid_structured_output",
            },
        ),
        ExecutionEvent(
            run_id="run_cli_runtime",
            event_type=EventType.ANSWER_DRAFTED,
            component="writer",
            summary="writer completed",
            payload={
                "backend": "llm",
                "model": "main-model",
                "fallback_reason": None,
            },
        ),
    ]
    fake_result = SimpleNamespace(
        run_id="run_cli_runtime",
        terminated_reason="no_new_evidence",
        answer_status=AnswerStatus.PARTIAL,
        events=events,
        plan=SimpleNamespace(answer_type="comparison", sub_questions=[]),
        verification=SimpleNamespace(
            is_sufficient=False,
            coverage_score=0.5,
            rationale_summary="Partial evidence.",
            conflicting_evidence_ids=[],
            corrective_queries=[],
        ),
        iteration=1,
        tool_call_count=2,
        evidence_ledger=SimpleNamespace(items=[]),
        token_usage=TokenUsage(total_tokens=24),
        latency_ms=10,
        unanswerable=False,
        final_answer=None,
    )
    _patch_ask_runtime(
        monkeypatch,
        api_key="configured-key",
        run_result=fake_result,
    )
    monkeypatch.setattr(
        "scholar_agent.llm.client.create_llm_client",
        lambda _config: object(),
    )
    result = runner.invoke(app, ["ask", "Compare Self-RAG versus CRAG"])

    assert result.exit_code == 0, result.output
    assert "requested=auto" in result.output
    assert "planner=deterministic" in result.output
    assert "model=fast-model" in result.output
    assert "fallback=invalid_structured_output" in result.output
    assert "writer=llm" in result.output
    assert "model=main-model" in result.output

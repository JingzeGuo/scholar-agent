"""Tests for the interview-focused public CLI surface."""

from __future__ import annotations

from typer.testing import CliRunner

from scholar_agent.cli import app

runner = CliRunner()


def test_public_help_keeps_interview_commands_and_removes_stage_artifacts() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("retrieve", "ask-naive", "ask", "evaluate", "demo"):
        assert f"│ {command}" in result.output
    for command in ("prototype", "ablate", "research"):
        assert f"│ {command}" not in result.output


def test_internal_diagnostic_commands_are_hidden_but_callable() -> None:
    for args in (
        ["research", "--help"],
        ["corpus", "summary", "--help"],
        ["graph", "stats", "--help"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0

    corpus_help = runner.invoke(app, ["corpus", "--help"])
    graph_help = runner.invoke(app, ["graph", "--help"])
    assert "summary" not in corpus_help.output
    assert "stats" not in graph_help.output


def test_removed_commands_are_not_callable() -> None:
    for command in ("prototype", "ablate"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code != 0
        assert "No such command" in result.output

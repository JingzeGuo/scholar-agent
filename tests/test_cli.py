from openai import OpenAIError
from typer.testing import CliRunner

import scholar_agent.cli as cli_module
from scholar_agent.cli import MissingAPIKeyError, app


def test_cli_exposes_only_three_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("ingest", "index", "ask"):
        assert command in result.output
    for removed in ("demo", "retrieve", "evaluate", "ablate", "corpus", "replay"):
        assert removed not in result.output


def test_cli_offline_and_online_failures_are_explicit(monkeypatch) -> None:
    runner = CliRunner()
    captured: list[bool] = []

    def offline_answer(question: str, offline: bool) -> str:
        captured.append(offline)
        return "offline answer"

    monkeypatch.setattr(cli_module, "_ask", offline_answer)
    offline = runner.invoke(app, ["ask", "question", "--offline"])
    assert offline.exit_code == 0
    assert captured == [True]

    monkeypatch.setattr(
        cli_module,
        "_ask",
        lambda question, offline: (_ for _ in ()).throw(MissingAPIKeyError()),
    )
    missing = runner.invoke(app, ["ask", "question"])
    assert missing.exit_code == 2
    assert "Run with --offline" in missing.output
    assert "Traceback" not in missing.output

    monkeypatch.setattr(
        cli_module,
        "_ask",
        lambda question, offline: (_ for _ in ()).throw(OpenAIError("down")),
    )
    unavailable = runner.invoke(app, ["ask", "question"])
    assert unavailable.exit_code == 1
    assert "LLM API unavailable" in unavailable.output

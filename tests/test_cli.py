from openai import OpenAIError
from typer.testing import CliRunner

import scholar_agent.cli as cli_module
from scholar_agent.cli import MissingAPIKeyError, app
from scholar_agent.indexes import ModelUnavailableError


def test_cli_exposes_only_three_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("ingest", "index", "ask"):
        assert command in result.output
    for removed in ("demo", "retrieve", "evaluate", "ablate", "corpus", "replay"):
        assert removed not in result.output


def test_cli_requires_online_dependencies_and_reports_failures(monkeypatch) -> None:
    runner = CliRunner()
    captured: list[str] = []

    def answer(question: str) -> str:
        captured.append(question)
        return "answer"

    monkeypatch.setattr(cli_module, "_ask", answer)
    success = runner.invoke(app, ["ask", "question"])
    assert success.exit_code == 0
    assert captured == ["question"]

    removed = runner.invoke(app, ["ask", "question", "--offline"])
    assert removed.exit_code == 2
    assert "No such option" in removed.output

    monkeypatch.setattr(
        cli_module,
        "_ask",
        lambda question: (_ for _ in ()).throw(MissingAPIKeyError()),
    )
    missing = runner.invoke(app, ["ask", "question"])
    assert missing.exit_code == 2
    assert "Set DEEPSEEK_API_KEY or OPENAI_API_KEY" in missing.output
    assert "Traceback" not in missing.output

    monkeypatch.setattr(
        cli_module,
        "_ask",
        lambda question: (_ for _ in ()).throw(OpenAIError("down")),
    )
    unavailable = runner.invoke(app, ["ask", "question"])
    assert unavailable.exit_code == 1
    assert "LLM API unavailable" in unavailable.output

    monkeypatch.setattr(
        cli_module,
        "_ask",
        lambda question: (_ for _ in ()).throw(ModelUnavailableError("model failed")),
    )
    model_failure = runner.invoke(app, ["ask", "question"])
    assert model_failure.exit_code == 1
    assert "model failed" in model_failure.output
    assert "Traceback" not in model_failure.output

    monkeypatch.setattr(
        cli_module,
        "build_all_indexes",
        lambda settings: (_ for _ in ()).throw(ModelUnavailableError("download failed")),
    )
    index_failure = runner.invoke(app, ["index"])
    assert index_failure.exit_code == 1
    assert "download failed" in index_failure.output
    assert "Traceback" not in index_failure.output

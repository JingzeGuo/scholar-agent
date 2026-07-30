from typer.testing import CliRunner

from scholar_agent.cli import app


def test_cli_exposes_only_four_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("ingest", "index", "ask", "demo"):
        assert command in result.output
    for removed in ("retrieve", "evaluate", "ablate", "corpus", "replay"):
        assert removed not in result.output

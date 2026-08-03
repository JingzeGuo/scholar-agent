from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch

from scholar_agent import config


def test_config_loads_dotenv_on_import(monkeypatch, tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "SCHOLAR_AGENT_TEST_DOTENV=loaded\n"
        "SCHOLAR_AGENT_MIN_RERANK_SCORE=0.25\n"
        "SCHOLAR_AGENT_MAX_RETRIES=2\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SCHOLAR_AGENT_TEST_DOTENV", raising=False)
    monkeypatch.delenv("SCHOLAR_AGENT_MIN_RERANK_SCORE", raising=False)
    monkeypatch.delenv("SCHOLAR_AGENT_MAX_RETRIES", raising=False)

    with patch("dotenv.main.find_dotenv", return_value=str(dotenv_path)):
        importlib.reload(config)

    assert os.getenv("SCHOLAR_AGENT_TEST_DOTENV") == "loaded"
    assert config.Settings.from_env().min_rerank_score == 0.25
    assert config.Settings.from_env().max_retries == 2

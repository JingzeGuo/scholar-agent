from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch

from scholar_agent import config


def test_config_loads_dotenv_on_import(monkeypatch, tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("SCHOLAR_AGENT_TEST_DOTENV=loaded\n", encoding="utf-8")
    monkeypatch.delenv("SCHOLAR_AGENT_TEST_DOTENV", raising=False)

    with patch("dotenv.main.find_dotenv", return_value=str(dotenv_path)):
        importlib.reload(config)

    assert os.getenv("SCHOLAR_AGENT_TEST_DOTENV") == "loaded"

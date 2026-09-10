from __future__ import annotations

import importlib
import os
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import pytest

import scholar_agent
import scholar_agent.llm as llm_module
from scholar_agent import config
from scholar_agent.config import Settings
from scholar_agent.llm import LLMClient


def test_package_version_comes_from_distribution_metadata() -> None:
    assert scholar_agent.__version__ == version("scholar-agent")


def test_config_loads_dotenv_on_import(monkeypatch, tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "SCHOLAR_AGENT_TEST_DOTENV=loaded\n"
        "SCHOLAR_AGENT_MIN_RERANK_SCORE=0.25\n"
        "SCHOLAR_AGENT_RETRIEVAL_MODE=fixed_hybrid\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SCHOLAR_AGENT_TEST_DOTENV", raising=False)
    monkeypatch.delenv("SCHOLAR_AGENT_MIN_RERANK_SCORE", raising=False)
    monkeypatch.delenv("SCHOLAR_AGENT_RETRIEVAL_MODE", raising=False)

    with patch("dotenv.main.find_dotenv", return_value=str(dotenv_path)):
        importlib.reload(config)

    assert os.getenv("SCHOLAR_AGENT_TEST_DOTENV") == "loaded"
    assert config.Settings.from_env().min_rerank_score == 0.25
    assert config.Settings.from_env().retrieval_mode == "fixed_hybrid"


def test_config_rejects_unknown_retrieval_mode() -> None:
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        Settings(retrieval_mode="automatic")


def test_config_rejects_unknown_recovery_mode() -> None:
    with pytest.raises(ValueError, match="Unknown recovery mode"):
        Settings(recovery_mode="automatic")


def test_controller_is_the_default_recovery_mode(monkeypatch) -> None:
    monkeypatch.delenv("SCHOLAR_AGENT_RECOVERY_MODE", raising=False)

    assert Settings().recovery_mode == "controller"
    assert Settings.from_env().recovery_mode == "controller"

    monkeypatch.setenv("SCHOLAR_AGENT_RECOVERY_MODE", "none")

    assert Settings.from_env().recovery_mode == "none"


def test_llm_client_uses_provider_specific_default_models(monkeypatch) -> None:
    clients: list[dict[str, str]] = []

    def fake_openai(**kwargs: str) -> object:
        clients.append(kwargs)
        return object()

    monkeypatch.setattr(llm_module, "OpenAI", fake_openai)
    monkeypatch.delenv("SCHOLAR_AGENT_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    deepseek = LLMClient.from_env(Settings())

    assert deepseek is not None
    assert deepseek.model == "deepseek-chat"
    assert clients[-1] == {
        "api_key": "deepseek-key",
        "base_url": "https://api.deepseek.com",
    }

    monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    openai = LLMClient.from_env(Settings())
    overridden = LLMClient.from_env(Settings(llm_model="custom-model"))

    assert openai is not None
    assert openai.model == "gpt-4.1-mini"
    assert overridden is not None
    assert overridden.model == "custom-model"
    assert clients[-1] == {"api_key": "openai-key"}

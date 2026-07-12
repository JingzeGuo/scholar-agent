"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scholar_agent.config import (
    AppConfig,
    BudgetsConfig,
    ChunkingConfig,
    EnvSettings,
    apply_env_overrides,
    load_config,
)


def test_load_default_config(default_config_path: Path, repo_root: Path) -> None:
    cfg = load_config(default_config_path, repo_root=repo_root, apply_env=False)
    assert isinstance(cfg, AppConfig)
    assert cfg.project.name == "scholar-agent"
    assert cfg.llm.main_model == "deepseek-v4-pro"
    assert cfg.llm.fast_model == "deepseek-v4-flash"
    assert cfg.budgets.max_tool_calls_per_research_pass == 4
    assert cfg.budgets.max_research_iterations_per_pass == 4
    assert cfg.chunking.target_tokens == 600
    assert cfg.paths.processed_dir.is_absolute()
    assert cfg.paths.processed_dir == (repo_root / "data" / "processed").resolve()


def test_env_overrides_api_key_and_budgets(default_config_path: Path, repo_root: Path) -> None:
    cfg = load_config(default_config_path, repo_root=repo_root, apply_env=False)
    env = EnvSettings(
        deepseek_api_key="sk-test-key",
        deepseek_main_model="deepseek-custom",
        scholar_max_tool_calls=2,
        scholar_max_research_iterations=3,
        scholar_max_corrective_iterations=1,
        scholar_log_level="DEBUG",
    )
    updated = apply_env_overrides(cfg, env)
    assert updated.llm.api_key == "sk-test-key"
    assert updated.llm.main_model == "deepseek-custom"
    assert updated.budgets.max_tool_calls_per_research_pass == 2
    assert updated.budgets.max_research_iterations_per_pass == 3
    assert updated.budgets.max_corrective_iterations == 1
    assert updated.logging.level == "DEBUG"
    # Original config object remains unchanged
    assert cfg.llm.api_key is None


def test_chunking_rejects_invalid_overlap() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(target_tokens=100, overlap_tokens=100, min_tokens=10)


def test_budgets_reject_non_positive() -> None:
    with pytest.raises(ValidationError):
        BudgetsConfig(max_tool_calls_per_research_pass=0)


def test_overrides_merge(default_config_path: Path, repo_root: Path) -> None:
    cfg = load_config(
        default_config_path,
        repo_root=repo_root,
        apply_env=False,
        overrides={"llm": {"temperature": 0.2}, "budgets": {"max_corrective_iterations": 5}},
    )
    assert cfg.llm.temperature == 0.2
    assert cfg.budgets.max_corrective_iterations == 5


def test_missing_config_file_raises(repo_root: Path, tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing, repo_root=repo_root, apply_env=False)

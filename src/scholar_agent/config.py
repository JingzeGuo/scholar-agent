"""Configuration loading and validation for ScholarAgent.

YAML configs under ``configs/`` provide defaults. Environment variables
(and optional ``.env``) can override LLM credentials and runtime budgets.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class ProjectConfig(BaseModel):
    name: str = "scholar-agent"
    version: str = "0.1.0"


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    papers_dir: Path = Path("data/papers")
    processed_dir: Path = Path("data/processed")
    indexes_dir: Path = Path("data/indexes")
    evaluation_dir: Path = Path("data/evaluation")
    corpus_manifest: Path = Path("data/corpus_manifest.jsonl")

    def resolve_against(self, root: Path) -> PathsConfig:
        """Return a copy with relative paths resolved against ``root``."""
        fields = self.model_dump()
        resolved: dict[str, Path] = {}
        for key, value in fields.items():
            path = Path(value)
            resolved[key] = path if path.is_absolute() else (root / path).resolve()
        return PathsConfig(**resolved)


class LLMConfig(BaseModel):
    provider: Literal["deepseek", "openai_compatible"] = "deepseek"
    base_url: str = "https://api.deepseek.com"
    main_model: str = "deepseek-v4-pro"
    fast_model: str = "deepseek-v4-flash"
    temperature: float = 0.0
    max_tokens: int = 4096
    request_timeout_s: float = 60.0
    max_retries: int = 3
    thinking_enabled: bool = False
    api_key: str | None = None

    @field_validator("temperature")
    @classmethod
    def _clamp_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return value


class BudgetsConfig(BaseModel):
    max_tool_calls_per_research_pass: int = Field(default=4, ge=1)
    max_research_iterations_per_pass: int = Field(default=4, ge=1)
    max_corrective_iterations: int = Field(default=3, ge=0)
    max_evidence_per_sub_question: int = Field(default=8, ge=1)
    max_total_tokens: int = Field(default=100_000, ge=1)
    max_latency_ms: int = Field(default=120_000, ge=1)


class DenseRetrievalConfig(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    top_k: int = Field(default=12, ge=1)
    collection_name: str = "scholar_chunks"


class SparseRetrievalConfig(BaseModel):
    top_k: int = Field(default=12, ge=1)


class FusionConfig(BaseModel):
    method: Literal["rrf"] = "rrf"
    k: int = Field(default=60, ge=1)
    fused_top_k: int = Field(default=20, ge=1)


class RerankerConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = Field(default=8, ge=1)


class GraphRetrievalConfig(BaseModel):
    max_hops: int = Field(default=2, ge=1, le=4)


class RetrievalConfig(BaseModel):
    dense: DenseRetrievalConfig = Field(default_factory=DenseRetrievalConfig)
    sparse: SparseRetrievalConfig = Field(default_factory=SparseRetrievalConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    graph: GraphRetrievalConfig = Field(default_factory=GraphRetrievalConfig)


class ChunkingConfig(BaseModel):
    target_tokens: int = Field(default=600, ge=50)
    overlap_tokens: int = Field(default=80, ge=0)
    min_tokens: int = Field(default=80, ge=1)
    encoding_name: str = "cl100k_base"
    allow_tokenizer_fallback: bool = False

    @field_validator("encoding_name")
    @classmethod
    def _non_empty_encoding(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("encoding_name must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _validate_overlap(self) -> ChunkingConfig:
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        if self.min_tokens > self.target_tokens:
            raise ValueError("min_tokens must be <= target_tokens")
        return self


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_logs: bool = False


class AppConfig(BaseModel):
    """Fully validated application configuration."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    config_path: Path | None = None
    repo_root: Path = REPO_ROOT


class EnvSettings(BaseSettings):
    """Environment overrides for secrets and hot-tunable budgets."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    deepseek_api_key: str | None = None
    deepseek_base_url: str | None = None
    deepseek_main_model: str | None = None
    deepseek_fast_model: str | None = None
    deepseek_thinking_enabled: bool | None = None

    openai_api_key: str | None = None

    scholar_max_tool_calls: int | None = None
    scholar_max_research_iterations: int | None = None
    scholar_max_corrective_iterations: int | None = None
    scholar_max_total_tokens: int | None = None
    scholar_request_timeout_s: float | None = None
    scholar_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_env_overrides(config: AppConfig, env: EnvSettings | None = None) -> AppConfig:
    """Apply environment variable overrides onto a loaded config."""
    env = env or EnvSettings()
    llm = config.llm.model_copy()
    budgets = config.budgets.model_copy()
    logging_cfg = config.logging.model_copy()

    api_key = env.deepseek_api_key or env.openai_api_key
    if api_key:
        llm.api_key = api_key
    if env.deepseek_base_url:
        llm.base_url = env.deepseek_base_url
    if env.deepseek_main_model:
        llm.main_model = env.deepseek_main_model
    if env.deepseek_fast_model:
        llm.fast_model = env.deepseek_fast_model
    if env.deepseek_thinking_enabled is not None:
        llm.thinking_enabled = env.deepseek_thinking_enabled
    if env.scholar_request_timeout_s is not None:
        llm.request_timeout_s = env.scholar_request_timeout_s

    if env.scholar_max_tool_calls is not None:
        budgets.max_tool_calls_per_research_pass = env.scholar_max_tool_calls
    if env.scholar_max_research_iterations is not None:
        budgets.max_research_iterations_per_pass = env.scholar_max_research_iterations
    if env.scholar_max_corrective_iterations is not None:
        budgets.max_corrective_iterations = env.scholar_max_corrective_iterations
    if env.scholar_max_total_tokens is not None:
        budgets.max_total_tokens = env.scholar_max_total_tokens

    if env.scholar_log_level is not None:
        logging_cfg.level = env.scholar_log_level

    return config.model_copy(
        update={
            "llm": llm,
            "budgets": budgets,
            "logging": logging_cfg,
        }
    )


def load_config(
    path: Path | str | None = None,
    *,
    repo_root: Path | None = None,
    apply_env: bool = True,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load, validate, and optionally env-override application configuration."""
    root = (repo_root or REPO_ROOT).resolve()
    config_path = Path(path) if path is not None else root / "configs" / "default.yaml"
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()

    raw = _load_yaml(config_path)
    if overrides:
        raw = _deep_merge(raw, overrides)

    # Strip keys not part of AppConfig nested models
    allowed = {
        "project",
        "paths",
        "llm",
        "budgets",
        "retrieval",
        "chunking",
        "logging",
    }
    filtered = {k: v for k, v in raw.items() if k in allowed}

    config = AppConfig.model_validate(
        {
            **filtered,
            "config_path": config_path,
            "repo_root": root,
        }
    )
    config = config.model_copy(update={"paths": config.paths.resolve_against(root)})

    if apply_env:
        config = apply_env_overrides(config)
    return config


@lru_cache(maxsize=4)
def get_config(path: str | None = None) -> AppConfig:
    """Cached config loader for process-wide defaults."""
    return load_config(path)

"""Optional live DeepSeek tests. Run with: pytest -m live

These tests are skipped unless DEEPSEEK_API_KEY (or OPENAI_API_KEY) is set.
"""

from __future__ import annotations

import os

import pytest
from scripts.deepseek_compatibility import (
    check_streaming,
    check_structured_json,
    check_tool_calling,
)

from scholar_agent.config import load_config
from scholar_agent.llm.client import ChatMessage, create_llm_client

pytestmark = pytest.mark.live


def _has_api_key() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))


@pytest.mark.skipif(not _has_api_key(), reason="No live API key configured")
def test_live_chat_completion(repo_root: object) -> None:
    from pathlib import Path

    root = Path(str(repo_root))
    config = load_config(root / "configs" / "default.yaml", repo_root=root)
    client = create_llm_client(config)
    response = client.chat(
        [ChatMessage(role="user", content="Reply with the single word: pong")],
        fast=True,
        max_tokens=16,
    )
    assert response.content
    assert response.content.strip()


@pytest.mark.skipif(not _has_api_key(), reason="No live API key configured")
def test_live_phase_zero_acceptance(repo_root: object) -> None:
    from pathlib import Path

    root = Path(str(repo_root))
    config = load_config(root / "configs" / "default.yaml", repo_root=root)
    client = create_llm_client(config)
    model = config.llm.fast_model

    results = [
        check_streaming(client, model),
        check_structured_json(client, model),
        check_tool_calling(client, model),
    ]
    failures = [f"{result.name}: {result.detail}" for result in results if not result.passed]
    assert not failures, "; ".join(failures)

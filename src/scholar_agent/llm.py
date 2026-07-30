"""Optional OpenAI-compatible LLM access; the default demo remains offline."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from scholar_agent.config import Settings


class LLMClient:
    """Tiny provider client used directly by agent nodes when a key is present."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    @classmethod
    def from_env(cls, settings: Settings) -> LLMClient | None:
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if deepseek_key:
            return cls(
                OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com"),
                settings.llm_model,
            )
        if openai_key:
            return cls(OpenAI(api_key=openai_key), settings.llm_model)
        return None

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def complete_json(self, prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value

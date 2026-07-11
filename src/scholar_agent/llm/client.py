"""OpenAI-compatible LLM client for DeepSeek and similar providers.

Phase 0 focuses on a thin, typed wrapper that:
- supports chat, streaming, tool calls, and structured JSON;
- records token usage when the provider returns it;
- never logs API keys;
- isolates provider-specific reasoning / thinking fields.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from scholar_agent.config import AppConfig, LLMConfig
from scholar_agent.llm.retry import call_with_retry
from scholar_agent.models import TokenUsage


class LLMClientConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.deepseek.com"
    main_model: str = "deepseek-v4-pro"
    fast_model: str = "deepseek-v4-flash"
    temperature: float = 0.0
    max_tokens: int = 4096
    request_timeout_s: float = 60.0
    max_retries: int = 3
    thinking_enabled: bool = False

    @classmethod
    def from_app_config(cls, config: AppConfig | LLMConfig) -> LLMClientConfig:
        llm = config.llm if isinstance(config, AppConfig) else config
        if not llm.api_key:
            raise ValueError(
                "LLM API key is not configured. Set DEEPSEEK_API_KEY in the environment "
                "or pass llm.api_key in configuration."
            )
        return cls(
            api_key=llm.api_key,
            base_url=llm.base_url,
            main_model=llm.main_model,
            fast_model=llm.fast_model,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            request_timeout_s=llm.request_timeout_s,
            max_retries=llm.max_retries,
            thinking_enabled=llm.thinking_enabled,
        )


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    # Provider-specific extras (e.g. reasoning content) — kept optional
    reasoning_content: str | None = None

    def to_openai_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_calls is not None:
            payload["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        return payload


class ChatResponse(BaseModel):
    content: str | None
    model: str
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_content: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _extract_reasoning(message: Any) -> str | None:
    """Pull provider-specific reasoning / thinking fields if present."""
    for attr in ("reasoning_content", "reasoning", "thinking"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    # Some SDKs put extras on model_extra / __dict__
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _usage_from_response(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


class LLMClient:
    """Thin typed wrapper around the OpenAI SDK for DeepSeek endpoints."""

    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout_s,
            # Retry explicitly below so its behavior is deterministic and testable.
            max_retries=0,
        )

    def _create_completion(self, **kwargs: Any) -> Any:
        return call_with_retry(
            lambda: self._client.chat.completions.create(**kwargs),
            max_attempts=self.config.max_retries + 1,
        )

    def _model_name(self, *, fast: bool = False, model: str | None = None) -> str:
        if model:
            return model
        return self.config.fast_model if fast else self.config.main_model

    def _extra_body(self) -> dict[str, Any] | None:
        """Provider-specific request fields (e.g. thinking mode toggles).

        DeepSeek V3/V4 thinking knobs vary by model/version. We pass a soft
        hint when thinking is disabled; unknown fields are typically ignored.
        """
        if self.config.thinking_enabled:
            return None
        return {"thinking": {"type": "disabled"}}

    def chat(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]],
        *,
        model: str | None = None,
        fast: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResponse:
        openai_messages = [
            m.to_openai_dict() if isinstance(m, ChatMessage) else m for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": self._model_name(fast=fast, model=model),
            "messages": openai_messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format

        body = extra_body if extra_body is not None else self._extra_body()
        if body:
            kwargs["extra_body"] = body

        response = self._create_completion(**kwargs)
        choice = response.choices[0]
        message = choice.message
        tool_calls: list[dict[str, Any]] = []
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        for tc in raw_tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )

        return ChatResponse(
            content=message.content,
            model=response.model or kwargs["model"],
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
            reasoning_content=_extract_reasoning(message),
            usage=_usage_from_response(response),
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def stream_chat(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]],
        *,
        model: str | None = None,
        fast: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        openai_messages = [
            m.to_openai_dict() if isinstance(m, ChatMessage) else m for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": self._model_name(fast=fast, model=model),
            "messages": openai_messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }
        body = self._extra_body()
        if body:
            kwargs["extra_body"] = body

        stream = self._create_completion(**kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content

    def chat_json(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]],
        *,
        model: str | None = None,
        fast: bool = True,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Request JSON object mode when the provider supports it."""
        return self.chat(
            messages,
            model=model,
            fast=fast,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )


def create_llm_client(config: AppConfig | LLMConfig | LLMClientConfig) -> LLMClient:
    if isinstance(config, LLMClientConfig):
        return LLMClient(config)
    return LLMClient(LLMClientConfig.from_app_config(config))

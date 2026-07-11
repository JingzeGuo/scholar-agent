"""LLM client utilities for DeepSeek / OpenAI-compatible providers."""

from scholar_agent.llm.client import LLMClient, LLMClientConfig, create_llm_client
from scholar_agent.llm.structured import StructuredOutputError, parse_structured_json

__all__ = [
    "LLMClient",
    "LLMClientConfig",
    "StructuredOutputError",
    "create_llm_client",
    "parse_structured_json",
]

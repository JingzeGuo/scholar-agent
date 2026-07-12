"""Explicit provider wiring for the optional RAGAS metrics."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from scholar_agent.config import LLMConfig
from scholar_agent.evaluation.answer_metrics import RagasEvaluator, try_ragas_scores
from scholar_agent.retrieval.embeddings import Embedder


class ScholarEmbeddings(Embeddings):  # type: ignore[misc]
    """Expose ScholarAgent's selected embedder through LangChain's interface."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)


def create_ragas_evaluator(
    llm_config: LLMConfig,
    embedder: Embedder,
) -> tuple[RagasEvaluator | None, dict[str, Any]]:
    """Build a RAGAS scorer without falling back to implicit provider defaults."""
    try:
        import datasets  # noqa: F401
        import ragas  # noqa: F401
    except Exception as exc:
        return None, {
            "available": False,
            "configured": False,
            "reason": f"optional dependencies unavailable: {type(exc).__name__}",
        }

    if not llm_config.api_key:
        return None, {
            "available": True,
            "configured": False,
            "reason": "DEEPSEEK_API_KEY/OPENAI_API_KEY is not configured",
        }

    llm = ChatOpenAI(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        model=llm_config.fast_model,
        temperature=0.0,
        max_retries=llm_config.max_retries,
        timeout=llm_config.request_timeout_s,
    )
    embeddings = ScholarEmbeddings(embedder)

    def score(**kwargs: Any) -> dict[str, float] | None:
        return try_ragas_scores(llm=llm, embeddings=embeddings, **kwargs)

    return score, {
        "available": True,
        "configured": True,
        "provider": llm_config.provider,
        "base_url": llm_config.base_url,
        "model": llm_config.fast_model,
        "embedding_model": embedder.model_name,
    }

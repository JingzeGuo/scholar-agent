#!/usr/bin/env python3
"""DeepSeek API compatibility spike (Phase 0).

Verifies against a live DeepSeek / OpenAI-compatible endpoint:

1. normal chat completion
2. streaming
3. structured JSON output
4. tool calling
5. multi-turn tool use
6. handling of reasoning / thinking-related response fields
7. malformed-JSON and transient provider retry behavior

Usage:
    cp .env.example .env   # set DEEPSEEK_API_KEY
    uv run python scripts/deepseek_compatibility.py

Exit code 0 only if all enabled checks pass. Live calls require an API key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow running without install: add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from scholar_agent.config import load_config  # noqa: E402
from scholar_agent.llm.client import ChatMessage, LLMClient, create_llm_client  # noqa: E402
from scholar_agent.llm.retry import call_with_retry  # noqa: E402
from scholar_agent.llm.structured import request_structured_json_with_retry  # noqa: E402
from scholar_agent.models import PrototypeDecision  # noqa: E402


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    latency_ms: int = 0
    skipped: bool = False


@dataclass
class CompatibilityReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> bool:
        return all(r.passed or r.skipped for r in self.results)

    def print_summary(self) -> None:
        print("\n=== DeepSeek Compatibility Report ===")
        for r in self.results:
            if r.skipped:
                status = "SKIP"
            elif r.passed:
                status = "PASS"
            else:
                status = "FAIL"
            print(f"[{status}] {r.name} ({r.latency_ms} ms) — {r.detail}")
        overall = "PASSED" if self.passed else "FAILED"
        print(f"\nOverall: {overall}")

    def write_json(self, path: Path, *, model: str | None = None) -> None:
        """Write a secret-free, machine-readable acceptance record."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "model": model,
            "passed": self.passed,
            "results": [asdict(result) for result in self.results],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _timed(fn: Any) -> tuple[Any, int]:
    start = time.perf_counter()
    value = fn()
    elapsed = int((time.perf_counter() - start) * 1000)
    return value, elapsed


def check_chat(client: LLMClient, model: str) -> CheckResult:
    def _run() -> str:
        response = client.chat(
            [
                ChatMessage(role="system", content="Reply briefly."),
                ChatMessage(role="user", content="Say hello in one short sentence."),
            ],
            model=model,
            max_tokens=64,
        )
        if not response.content or not response.content.strip():
            raise AssertionError("empty content")
        return f"model={response.model}; finish={response.finish_reason}; chars={len(response.content)}"

    try:
        detail, ms = _timed(_run)
        return CheckResult("chat_completion", True, detail, ms)
    except Exception as exc:  # noqa: BLE001 — report all provider failures
        return CheckResult("chat_completion", False, f"{type(exc).__name__}: {exc}")


def check_streaming(client: LLMClient, model: str) -> CheckResult:
    def _run() -> str:
        chunks: list[str] = []
        for piece in client.stream_chat(
            [ChatMessage(role="user", content="Count from 1 to 3, separated by spaces.")],
            model=model,
            max_tokens=32,
        ):
            chunks.append(piece)
        text = "".join(chunks).strip()
        if not text:
            raise AssertionError("no streamed tokens")
        return f"chunks={len(chunks)}; text={text[:80]!r}"

    try:
        detail, ms = _timed(_run)
        return CheckResult("streaming", True, detail, ms)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("streaming", False, f"{type(exc).__name__}: {exc}")


def check_structured_json(client: LLMClient, model: str) -> CheckResult:
    def _run() -> str:
        def request() -> str:
            response = client.chat_json(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "Return JSON with keys action (retrieve|verify|finish), "
                            "reason (string), need_more_evidence (boolean)."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=(
                            "We have no evidence yet for a literature question. "
                            "Decide next action."
                        ),
                    ),
                ],
                model=model,
                max_tokens=128,
            )
            return response.content or ""

        decision = request_structured_json_with_retry(request, PrototypeDecision, max_attempts=2)
        return f"action={decision.action}; need_more={decision.need_more_evidence}"

    try:
        detail, ms = _timed(_run)
        return CheckResult("structured_json", True, detail, ms)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("structured_json", False, f"{type(exc).__name__}: {exc}")


_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "dense_search",
            "description": "Semantic dense retrieval over paper chunks",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    }
]


def check_tool_calling(client: LLMClient, model: str) -> CheckResult:
    def _run() -> str:
        response = client.chat(
            [
                ChatMessage(
                    role="system",
                    content="Use tools when you need corpus evidence. Prefer dense_search.",
                ),
                ChatMessage(
                    role="user",
                    content="Find papers about Self-RAG using the dense_search tool.",
                ),
            ],
            model=model,
            tools=_TOOLS,
            tool_choice={"type": "function", "function": {"name": "dense_search"}},
            max_tokens=256,
        )
        if not response.has_tool_calls:
            raise AssertionError("provider returned no tool call")
        names = [tc["function"]["name"] for tc in response.tool_calls]
        if any(name != "dense_search" for name in names):
            raise AssertionError(f"unexpected tool names: {names}")
        for tool_call in response.tool_calls:
            arguments = json.loads(tool_call["function"]["arguments"])
            if not isinstance(arguments.get("query"), str) or not arguments["query"].strip():
                raise AssertionError("tool call is missing a non-empty query")
        return f"tool_calls={names}; finish={response.finish_reason}"

    try:
        detail, ms = _timed(_run)
        return CheckResult("tool_calling", True, detail, ms)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("tool_calling", False, f"{type(exc).__name__}: {exc}")


def check_multi_turn_tools(client: LLMClient, model: str) -> CheckResult:
    def _run() -> str:
        messages: list[ChatMessage | dict[str, Any]] = [
            ChatMessage(
                role="system",
                content="Use dense_search, then answer briefly from the tool result.",
            ),
            ChatMessage(role="user", content="What is Self-RAG? Use dense_search first."),
        ]
        first = client.chat(
            messages,
            model=model,
            tools=_TOOLS,
            tool_choice={"type": "function", "function": {"name": "dense_search"}},
            max_tokens=256,
        )
        if not first.has_tool_calls:
            raise AssertionError("first turn returned no tool call")

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": first.content,
            "tool_calls": first.tool_calls,
        }
        messages.append(assistant_msg)
        for tc in first.tool_calls:
            fake_result = json.dumps(
                {
                    "results": [
                        {
                            "chunk_id": "chunk_demo_1",
                            "paper_id": "paper_self_rag",
                            "page_start": 1,
                            "page_end": 2,
                            "text": "Self-RAG retrieves on demand and critiques generation.",
                            "score": 0.93,
                        }
                    ]
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": fake_result,
                }
            )

        second = client.chat(
            messages,
            model=model,
            tools=_TOOLS,
            max_tokens=256,
        )
        content = (second.content or "").strip()
        if not content:
            raise AssertionError("empty second-turn content")
        return f"second_turn_chars={len(content)}; finish={second.finish_reason}"

    try:
        detail, ms = _timed(_run)
        return CheckResult("multi_turn_tool_use", True, detail, ms)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("multi_turn_tool_use", False, f"{type(exc).__name__}: {exc}")


def check_reasoning_fields(client: LLMClient, model: str) -> CheckResult:
    """Probe whether reasoning/thinking fields appear and are handled cleanly."""

    def _run() -> str:
        response = client.chat(
            [
                ChatMessage(
                    role="user",
                    content="In one sentence, what is retrieval-augmented generation?",
                )
            ],
            model=model,
            max_tokens=128,
        )
        has_reasoning = bool(response.reasoning_content)
        content_ok = bool(response.content and response.content.strip())
        if not content_ok:
            raise AssertionError("missing content after reasoning field extraction")
        return (
            f"has_reasoning_field={has_reasoning}; "
            f"content_chars={len(response.content or '')}; "
            f"reasoning_chars={len(response.reasoning_content or '')}"
        )

    try:
        detail, ms = _timed(_run)
        return CheckResult("reasoning_fields", True, detail, ms)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("reasoning_fields", False, f"{type(exc).__name__}: {exc}")


def check_thinking_modes(client: LLMClient, model: str) -> CheckResult:
    """Attempt non-thinking mode (default) and optionally thinking mode."""

    def _run() -> str:
        # Non-thinking (client default when thinking_enabled=False)
        normal = client.chat(
            [ChatMessage(role="user", content="Reply with the word OK.")],
            model=model,
            max_tokens=16,
            extra_body={"thinking": {"type": "disabled"}},
        )
        if not normal.content:
            raise AssertionError("non-thinking response empty")

        thinking = client.chat(
            [ChatMessage(role="user", content="Reply with the word OK.")],
            model=model,
            max_tokens=64,
            extra_body={"thinking": {"type": "enabled"}},
        )
        if not thinking.content and not thinking.reasoning_content:
            raise AssertionError("thinking response contained neither answer nor reasoning field")
        return (
            "non_thinking_ok=True; "
            f"thinking_content_chars={len(thinking.content or '')}; "
            f"reasoning_chars={len(thinking.reasoning_content or '')}"
        )

    try:
        detail, ms = _timed(_run)
        return CheckResult("thinking_modes", True, detail, ms)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("thinking_modes", False, f"{type(exc).__name__}: {exc}")


class SyntheticRateLimitError(Exception):
    """Offline probe error with the same surface as a provider HTTP 429."""

    status_code = 429


def check_retry_behavior(_client: LLMClient, _model: str) -> CheckResult:
    """Verify malformed-JSON and rate-limit retry paths without causing paid failures."""

    def _run() -> str:
        json_attempts = 0

        def malformed_then_valid() -> str:
            nonlocal json_attempts
            json_attempts += 1
            if json_attempts == 1:
                return "not-json"
            return '{"action":"finish","reason":"ok","need_more_evidence":false}'

        decision = request_structured_json_with_retry(
            malformed_then_valid,
            PrototypeDecision,
            max_attempts=2,
        )

        rate_limit_attempts = 0

        def rate_limited_then_ok() -> str:
            nonlocal rate_limit_attempts
            rate_limit_attempts += 1
            if rate_limit_attempts == 1:
                raise SyntheticRateLimitError("synthetic 429")
            return "ok"

        result = call_with_retry(
            rate_limited_then_ok,
            max_attempts=2,
            base_delay_s=0,
        )
        if decision.action != "finish" or result != "ok":
            raise AssertionError("retry probes returned unexpected values")
        return f"malformed_json_attempts={json_attempts}; rate_limit_attempts={rate_limit_attempts}"

    try:
        detail, ms = _timed(_run)
        return CheckResult("retry_behavior", True, detail, ms)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("retry_behavior", False, f"{type(exc).__name__}: {exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "outputs" / "deepseek_compatibility.json",
        help="Path for the secret-free JSON acceptance report.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_dotenv(REPO_ROOT / ".env")
    report = CompatibilityReport()

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "No DEEPSEEK_API_KEY (or OPENAI_API_KEY) set.\n"
            "Copy .env.example to .env and add your key, then re-run.\n"
            "Offline unit tests cover structured parsing and the prototype loop."
        )
        report.add(
            CheckResult(
                "api_key",
                passed=False,
                detail="missing API key",
                skipped=False,
            )
        )
        report.print_summary()
        report.write_json(args.report)
        return 1

    try:
        config = load_config(REPO_ROOT / "configs" / "default.yaml")
        client = create_llm_client(config)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to create client: {exc}")
        traceback.print_exc()
        return 1

    # Prefer the fast model for the spike to limit cost/latency
    model = config.llm.fast_model
    print(f"Provider base_url={config.llm.base_url}")
    print(f"Using model={model}")
    print(f"thinking_enabled={config.llm.thinking_enabled}")

    checks = [
        check_chat,
        check_streaming,
        check_structured_json,
        check_tool_calling,
        check_multi_turn_tools,
        check_reasoning_fields,
        check_thinking_modes,
        check_retry_behavior,
    ]
    for check in checks:
        print(f"Running {check.__name__}...")
        report.add(check(client, model))

    report.print_summary()
    report.write_json(args.report, model=model)
    print(f"Report: {args.report}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

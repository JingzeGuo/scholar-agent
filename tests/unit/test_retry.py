"""Tests for explicit provider retry behavior."""

from __future__ import annotations

import pytest

from scholar_agent.llm.retry import call_with_retry, is_retryable_provider_error


class FakeRateLimitError(Exception):
    status_code = 429


def test_rate_limit_is_retried() -> None:
    attempts = 0
    delays: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise FakeRateLimitError("try later")
        return "ok"

    result = call_with_retry(
        operation,
        max_attempts=3,
        base_delay_s=0.5,
        sleep=delays.append,
    )

    assert result == "ok"
    assert attempts == 3
    assert delays == [0.5, 1.0]


def test_non_retryable_error_fails_immediately() -> None:
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid request")

    with pytest.raises(ValueError, match="invalid request"):
        call_with_retry(operation, max_attempts=3, base_delay_s=0)

    assert attempts == 1


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_retryable_http_statuses(status_code: int) -> None:
    error = RuntimeError("provider error")
    error.status_code = status_code  # type: ignore[attr-defined]
    assert is_retryable_provider_error(error)

"""Tests for explicit provider retry behavior."""

from __future__ import annotations

import random

import pytest

from scholar_agent.llm.retry import (
    call_with_retry,
    compute_backoff_delay,
    is_retryable_provider_error,
)


class FakeRateLimitError(Exception):
    status_code = 429


class FakeAuthError(Exception):
    status_code = 401


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
        jitter=0.0,
        sleep=delays.append,
    )

    assert result == "ok"
    assert attempts == 3
    assert delays == [0.5, 1.0]


def test_backoff_includes_jitter() -> None:
    rng = random.Random(0)
    delays = [compute_backoff_delay(1, base_delay_s=1.0, jitter=0.5, rng=rng) for _ in range(20)]
    assert min(delays) >= 0.5
    assert max(delays) <= 1.0
    assert len(set(round(d, 6) for d in delays)) > 1


def test_non_retryable_error_fails_immediately() -> None:
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid request")

    with pytest.raises(ValueError, match="invalid request"):
        call_with_retry(operation, max_attempts=3, base_delay_s=0, jitter=0.0)

    assert attempts == 1


def test_auth_error_not_retried() -> None:
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise FakeAuthError("bad key")

    with pytest.raises(FakeAuthError):
        call_with_retry(operation, max_attempts=4, base_delay_s=0.1, jitter=0.0)

    assert attempts == 1
    assert is_retryable_provider_error(FakeAuthError("x")) is False


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_retryable_http_statuses(status_code: int) -> None:
    error = RuntimeError("provider error")
    error.status_code = status_code  # type: ignore[attr-defined]
    assert is_retryable_provider_error(error)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_non_retryable_http_statuses(status_code: int) -> None:
    error = RuntimeError("client error")
    error.status_code = status_code  # type: ignore[attr-defined]
    assert is_retryable_provider_error(error) is False

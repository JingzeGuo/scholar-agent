"""Small, testable retry policy for transient provider failures."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def is_retryable_provider_error(exc: Exception) -> bool:
    """Return whether an exception represents a transient provider failure."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or isinstance(status_code, int) and status_code >= 500:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def call_with_retry(
    operation: Callable[[], T],
    *,
    max_attempts: int,
    base_delay_s: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``operation`` with bounded exponential backoff for transient errors."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == max_attempts or not is_retryable_provider_error(exc):
                raise
            sleep(base_delay_s * 2 ** (attempt - 1))
    raise AssertionError("retry loop terminated unexpectedly")

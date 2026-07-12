"""Small, testable retry policy for transient provider failures."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# Permanent client errors that must never be retried as if they were transient.
_NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 422})


def is_retryable_provider_error(exc: Exception) -> bool:
    """Return whether an exception represents a transient provider failure."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code in _NON_RETRYABLE_STATUS:
            return False
        if status_code == 429 or status_code >= 500:
            return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__
    if name in {"AuthenticationError", "PermissionDeniedError", "BadRequestError", "NotFoundError"}:
        return False
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def compute_backoff_delay(
    attempt: int,
    *,
    base_delay_s: float = 0.25,
    jitter: float = 0.25,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff with optional full-jitter fraction in ``[0, jitter]``.

    ``attempt`` is 1-based (first retry after the first failure uses attempt=1).
    """
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    if base_delay_s < 0:
        raise ValueError("base_delay_s must be >= 0")
    if not 0.0 <= jitter <= 1.0:
        raise ValueError("jitter must be between 0.0 and 1.0")
    exp = float(base_delay_s * (2 ** (attempt - 1)))
    if jitter <= 0 or exp <= 0:
        return exp
    generator = rng if rng is not None else random.Random()
    # Full jitter on the exponential delay: uniform in [(1-jitter)*exp, exp]
    low = exp * (1.0 - jitter)
    return float(generator.uniform(low, exp))


def call_with_retry(
    operation: Callable[[], T],
    *,
    max_attempts: int,
    base_delay_s: float = 0.25,
    jitter: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Run ``operation`` with bounded exponential backoff for transient errors.

    Permanent errors (validation, auth, 4xx other than 429) are not retried.
    Retries never run indefinitely: ``max_attempts`` is a hard cap.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts or not is_retryable_provider_error(exc):
                raise
            delay = compute_backoff_delay(
                attempt,
                base_delay_s=base_delay_s,
                jitter=jitter,
                rng=rng,
            )
            sleep(delay)
    raise AssertionError(f"retry loop terminated unexpectedly: {last_error}")

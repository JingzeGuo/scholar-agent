"""Structured logging helpers.

Logs must never include API keys or raw secrets. Execution traces for the
agent system should use dedicated event models rather than dumping chain-of-thought.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from scholar_agent.config import AppConfig, LoggingConfig

_STANDARD_LOG_RECORD_KEYS = frozenset(
    {
        *logging.makeLogRecord({}).__dict__.keys(),
        "asctime",
        "message",
    }
)


class _JsonFormatter(logging.Formatter):
    def __init__(self, *, secrets: Sequence[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _sanitize_log_value(record.getMessage(), secrets=self._secrets),
        }
        if record.exc_info:
            payload["exc_info"] = _sanitize_log_value(
                self.formatException(record.exc_info),
                secrets=self._secrets,
            )
        if record.stack_info:
            payload["stack_info"] = _sanitize_log_value(
                self.formatStack(record.stack_info),
                secrets=self._secrets,
            )
        # Preserve structured ``extra={...}`` fields while sanitizing nested
        # values. Standard LogRecord implementation details are intentionally
        # excluded from the JSON record.
        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_KEYS or key.startswith("_"):
                continue
            safe_key = sanitize_for_log(str(key), secrets=list(self._secrets))
            payload[safe_key] = _sanitize_log_value(value, secrets=self._secrets)
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)


class _SanitizingFormatter(logging.Formatter):
    """Sanitize the complete rendered line, including exception tracebacks."""

    def __init__(self, *args: Any, secrets: Sequence[str] = (), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._secrets = tuple(secret for secret in secrets if secret)

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return sanitize_for_log(rendered, secrets=list(self._secrets))


def setup_logging(config: LoggingConfig | AppConfig | None = None) -> None:
    """Configure root logging once for CLI / scripts."""
    secrets: tuple[str, ...] = ()
    if isinstance(config, AppConfig):
        log_cfg = config.logging
        if config.llm.api_key:
            secrets = (config.llm.api_key,)
    elif isinstance(config, LoggingConfig):
        log_cfg = config
    else:
        log_cfg = LoggingConfig()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_cfg.level)

    handler = logging.StreamHandler(sys.stderr)
    if log_cfg.json_logs:
        handler.setFormatter(_JsonFormatter(secrets=secrets))
    else:
        handler.setFormatter(
            _SanitizingFormatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
                secrets=secrets,
            )
        )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?i)(api[_-]?key|authorization|bearer|token)\s*[:=]\s*['\"]?([^\s'\"]+)"),
        r"\1=***REDACTED***",
    ),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "***REDACTED***"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b"), "Bearer ***REDACTED***"),
]


def sanitize_for_log(value: str, *, secrets: list[str] | None = None) -> str:
    """Redact known secrets and common credential patterns from a string before logging."""
    redacted = value
    for secret in secrets or []:
        if secret:
            redacted = redacted.replace(secret, "***REDACTED***")
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _sanitize_log_value(value: Any, *, secrets: Sequence[str]) -> Any:
    """Return a JSON-safe, recursively sanitized structured-log value."""
    explicit = list(secrets)
    if isinstance(value, str):
        return sanitize_for_log(value, secrets=explicit)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else sanitize_for_log(str(value), secrets=explicit)
    if isinstance(value, Mapping):
        return {
            sanitize_for_log(str(key), secrets=explicit): _sanitize_log_value(
                nested,
                secrets=secrets,
            )
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize_log_value(item, secrets=secrets) for item in value]
    return sanitize_for_log(str(value), secrets=explicit)

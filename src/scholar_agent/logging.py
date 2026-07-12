"""Structured logging helpers.

Logs must never include API keys or raw secrets. Execution traces for the
agent system should use dedicated event models rather than dumping chain-of-thought.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

from scholar_agent.config import AppConfig, LoggingConfig


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key in ("run_id", "event_type", "component"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(config: LoggingConfig | AppConfig | None = None) -> None:
    """Configure root logging once for CLI / scripts."""
    if isinstance(config, AppConfig):
        log_cfg = config.logging
    elif isinstance(config, LoggingConfig):
        log_cfg = config
    else:
        log_cfg = LoggingConfig()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_cfg.level)

    handler = logging.StreamHandler(sys.stderr)
    if log_cfg.json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
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

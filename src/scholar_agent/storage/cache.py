"""Deterministic disk cache for expensive, pure offline computations.

Cache design (Phase 10):
- Keys are SHA-256 of namespace + schema version + canonical payload JSON.
- Entries are versioned; mismatched schema versions miss.
- Writes are atomic (temp file + replace).
- Corrupt or unreadable entries are treated as misses and removed.
- Values must be JSON-serializable; secrets must never be stored.
- Does not cache mutable workflow state (evidence ledgers, live runs).

Invalidation:
- schema_version change
- payload change (source text, config knobs included in key)
- explicit delete / clear
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

from scholar_agent.ids import content_hash

logger = logging.getLogger(__name__)

T = TypeVar("T")

CACHE_SCHEMA_VERSION = "1"
DEFAULT_NAMESPACE = "default"


class CacheError(RuntimeError):
    """Raised for non-recoverable cache configuration problems."""


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    corruptions: int = 0
    invalidations: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "corruptions": self.corruptions,
            "invalidations": self.invalidations,
        }


@dataclass
class DiskCache:
    """File-backed JSON cache with deterministic keys and safe writes."""

    root: Path
    namespace: str = DEFAULT_NAMESPACE
    schema_version: str = CACHE_SCHEMA_VERSION
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.namespace or not self.namespace.replace("_", "").replace("-", "").isalnum():
            raise CacheError(
                f"invalid cache namespace {self.namespace!r}; "
                "use alphanumeric characters, '_' or '-'"
            )
        self.root.mkdir(parents=True, exist_ok=True)

    def make_key(self, payload: Any) -> str:
        """Build a deterministic key from a JSON-serializable payload."""
        canonical = _canonical_json(
            {
                "namespace": self.namespace,
                "schema_version": self.schema_version,
                "payload": payload,
            }
        )
        return content_hash(canonical)

    def path_for_key(self, key: str) -> Path:
        if len(key) < 4 or not all(c in "0123456789abcdef" for c in key.lower()):
            raise CacheError(f"invalid cache key format: {key!r}")
        # Shard by first two hex chars to keep directories shallow
        return self.root / self.namespace / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self.path_for_key(key)
        if not path.is_file():
            self.stats.misses += 1
            logger.debug("cache_miss namespace=%s key=%s", self.namespace, key[:12])
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.stats.corruptions += 1
            self.stats.misses += 1
            logger.warning(
                "cache_corrupt namespace=%s key=%s err=%s",
                self.namespace,
                key[:12],
                type(exc).__name__,
            )
            _safe_unlink(path)
            return None
        if not isinstance(raw, dict):
            self.stats.corruptions += 1
            self.stats.misses += 1
            _safe_unlink(path)
            return None
        if raw.get("schema_version") != self.schema_version:
            self.stats.invalidations += 1
            self.stats.misses += 1
            logger.debug(
                "cache_schema_miss namespace=%s key=%s found=%s expected=%s",
                self.namespace,
                key[:12],
                raw.get("schema_version"),
                self.schema_version,
            )
            _safe_unlink(path)
            return None
        if raw.get("namespace") != self.namespace:
            self.stats.invalidations += 1
            self.stats.misses += 1
            _safe_unlink(path)
            return None
        if "value" not in raw:
            self.stats.corruptions += 1
            self.stats.misses += 1
            _safe_unlink(path)
            return None
        self.stats.hits += 1
        logger.debug("cache_hit namespace=%s key=%s", self.namespace, key[:12])
        return raw["value"]

    def set(self, key: str, value: Any) -> Path:
        """Store a JSON-serializable value under ``key``."""
        path = self.path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "namespace": self.namespace,
            "schema_version": self.schema_version,
            "key": key,
            "value": value,
        }
        # Validate serializability before writing
        payload = _canonical_json(record)
        _atomic_write_text(path, payload + "\n")
        self.stats.stores += 1
        logger.debug("cache_store namespace=%s key=%s", self.namespace, key[:12])
        return path

    def delete(self, key: str) -> bool:
        path = self.path_for_key(key)
        if path.is_file():
            _safe_unlink(path)
            self.stats.invalidations += 1
            return True
        return False

    def clear(self) -> int:
        """Remove all entries for this namespace. Returns number of files removed."""
        base = self.root / self.namespace
        removed = 0
        if not base.exists():
            return 0
        for path in base.rglob("*.json"):
            _safe_unlink(path)
            removed += 1
        self.stats.invalidations += removed
        return removed

    def get_or_set(
        self,
        payload: Any,
        compute: Callable[[], T],
        *,
        serializer: Callable[[T], Any] | None = None,
        deserializer: Callable[[Any], T] | None = None,
    ) -> T:
        """Return cached value for ``payload`` or compute, store, and return it."""
        key = self.make_key(payload)
        cached = self.get(key)
        if cached is not None:
            if deserializer is not None:
                return deserializer(cached)
            return cast(T, cached)
        value = compute()
        to_store: Any = serializer(value) if serializer is not None else value
        self.set(key, to_store)
        return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        _safe_unlink(tmp_path)
        raise


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("cache_unlink_failed path=%s err=%s", path, type(exc).__name__)

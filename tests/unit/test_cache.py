"""Disk cache hit/miss/invalidation/corruption tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.storage.cache import CacheError, DiskCache


def test_cache_hit_and_miss(tmp_path: Path) -> None:
    cache = DiskCache(root=tmp_path / "cache", namespace="extract")
    key = cache.make_key({"chunk": "c1", "text_hash": "abc"})
    assert cache.get(key) is None
    assert cache.stats.misses == 1

    cache.set(key, {"relations": 2})
    assert cache.get(key) == {"relations": 2}
    assert cache.stats.hits == 1
    assert cache.stats.stores == 1


def test_cache_key_deterministic(tmp_path: Path) -> None:
    cache = DiskCache(root=tmp_path / "cache")
    payload = {"b": 2, "a": [1, 3]}
    key = cache.make_key(payload)
    assert key == cache.make_key({"a": [1, 3], "b": 2})
    assert len(key) == 64


def test_schema_version_invalidates(tmp_path: Path) -> None:
    cache_v1 = DiskCache(root=tmp_path / "cache", schema_version="v1")
    key = cache_v1.make_key({"x": 1})
    cache_v1.set(key, {"ok": True})

    cache_v2 = DiskCache(root=tmp_path / "cache", schema_version="v2")
    # Same payload but different schema → different key; also old file wrong schema
    assert cache_v2.get(key) is None
    assert cache_v2.stats.invalidations >= 1 or cache_v2.stats.misses >= 1


def test_corrupt_entry_treated_as_miss(tmp_path: Path) -> None:
    cache = DiskCache(root=tmp_path / "cache", namespace="demo")
    key = cache.make_key({"k": 1})
    path = cache.path_for_key(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")

    assert cache.get(key) is None
    assert cache.stats.corruptions == 1
    assert not path.exists()


def test_embedded_key_mismatch_treated_as_corruption(tmp_path: Path) -> None:
    cache = DiskCache(root=tmp_path / "cache", namespace="demo")
    key = cache.make_key({"k": 1})
    path = cache.set(key, {"safe": True})
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["key"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert cache.get(key) is None
    assert cache.stats.corruptions == 1
    assert not path.exists()


def test_non_finite_payload_is_rejected(tmp_path: Path) -> None:
    cache = DiskCache(root=tmp_path / "cache")

    with pytest.raises(ValueError, match="JSON compliant"):
        cache.make_key({"score": float("nan")})


def test_get_or_set_computes_once(tmp_path: Path) -> None:
    cache = DiskCache(root=tmp_path / "cache")
    calls = {"n": 0}

    def compute() -> dict[str, int]:
        calls["n"] += 1
        return {"value": 42}

    first = cache.get_or_set({"input": "x"}, compute)
    second = cache.get_or_set({"input": "x"}, compute)
    assert first == second == {"value": 42}
    assert calls["n"] == 1
    assert cache.stats.hits == 1
    assert cache.stats.stores == 1


def test_clear_namespace(tmp_path: Path) -> None:
    cache = DiskCache(root=tmp_path / "cache", namespace="ns")
    key = cache.make_key({"a": 1})
    cache.set(key, 1)
    assert cache.clear() >= 1
    assert cache.get(key) is None


def test_invalid_namespace_rejected(tmp_path: Path) -> None:
    with pytest.raises(CacheError):
        DiskCache(root=tmp_path, namespace="../evil")


def test_atomic_write_creates_valid_json(tmp_path: Path) -> None:
    cache = DiskCache(root=tmp_path / "cache")
    key = cache.make_key({"payload": "ok"})
    path = cache.set(key, [1, 2, 3])
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["value"] == [1, 2, 3]
    assert raw["schema_version"] == cache.schema_version

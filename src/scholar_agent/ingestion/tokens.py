"""Token counting helpers with a deterministic, offline-safe fallback."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import tiktoken


class TokenEncoding(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class TokenizerUnavailableError(RuntimeError):
    """Raised when reproducible ingestion cannot load its configured tokenizer."""


class _LocalReversibleEncoding:
    """Small deterministic tokenizer used when a tiktoken vocabulary is absent.

    It groups Unicode words, punctuation, and their leading whitespace.  Each
    token is reversibly packed into a Python integer, so chunk splitting keeps
    text intact without a vocabulary download or a mutable process-wide token
    dictionary.
    """

    _pattern = re.compile(r"\s*\w+|\s*[^\w\s]+|\s+", re.UNICODE)

    def encode(self, text: str) -> list[int]:
        return [
            int.from_bytes(b"\x01" + match.group(0).encode("utf-8"), "big")
            for match in self._pattern.finditer(text)
        ]

    def decode(self, tokens: list[int]) -> str:
        parts: list[str] = []
        for token in tokens:
            if token <= 0:
                raise ValueError("invalid local token id")
            packed = token.to_bytes(max(1, (token.bit_length() + 7) // 8), "big")
            if not packed.startswith(b"\x01"):
                raise ValueError("token was not produced by the local encoding")
            parts.append(packed[1:].decode("utf-8"))
        return "".join(parts)


# tiktoken intentionally fetches these vocabularies on first use.  The default
# test/ingestion path must not make an implicit network request.  When a valid
# cache is already present we still use the requested production vocabulary.
_TIKTOKEN_ASSETS: dict[str, tuple[str, str]] = {
    "cl100k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    ),
}


def _cached_asset_is_valid(name: str) -> bool:
    asset = _TIKTOKEN_ASSETS.get(name)
    if asset is None:
        return True
    url, expected_sha256 = asset
    cache_dir = os.getenv("TIKTOKEN_CACHE_DIR") or os.getenv("DATA_GYM_CACHE_DIR")
    if cache_dir == "":
        return False
    root = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "data-gym-cache"
    path = root / hashlib.sha1(url.encode("utf-8")).hexdigest()
    try:
        payload = path.read_bytes()
    except OSError:
        return False
    return hashlib.sha256(payload).hexdigest() == expected_sha256


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


@lru_cache(maxsize=16)
def _get_encoding(
    name: str,
    *,
    allow_download: bool,
    allow_fallback: bool,
    cache_identity: str,
) -> TokenEncoding:
    del cache_identity  # part of the cache key; tiktoken reads the environment
    has_remote_asset = name in _TIKTOKEN_ASSETS
    if has_remote_asset and not allow_download and not _cached_asset_is_valid(name):
        if allow_fallback:
            return _LocalReversibleEncoding()
        raise TokenizerUnavailableError(
            f"Tokenizer asset for {name!r} is unavailable. Set "
            "TIKTOKEN_CACHE_DIR=.cache/tiktoken and run once with "
            "SCHOLAR_ALLOW_TOKENIZER_DOWNLOAD=1, or explicitly opt into the "
            "non-canonical fallback with --allow-tokenizer-fallback."
        )
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # noqa: BLE001 - converted to an actionable error
        if has_remote_asset and allow_fallback:
            return _LocalReversibleEncoding()
        raise TokenizerUnavailableError(f"Unable to load tokenizer {name!r}") from exc


def get_encoding(
    name: str = "cl100k_base",
    *,
    allow_fallback: bool = True,
) -> TokenEncoding:
    """Load an encoding without an implicit network request.

    General-purpose helpers keep an offline deterministic fallback for unit
    tests.  Canonical ingestion calls :func:`require_encoding` first, where the
    fallback is disabled unless the operator explicitly opts in.
    """
    allow_download = _truthy_env("SCHOLAR_ALLOW_TOKENIZER_DOWNLOAD")
    explicit_fallback = _truthy_env("SCHOLAR_ALLOW_TOKENIZER_FALLBACK")
    cache_identity = "|".join(
        (
            os.getenv("TIKTOKEN_CACHE_DIR", ""),
            os.getenv("DATA_GYM_CACHE_DIR", ""),
        )
    )
    return _get_encoding(
        name,
        allow_download=allow_download,
        allow_fallback=allow_fallback or explicit_fallback,
        cache_identity=cache_identity,
    )


def encoding_backend(encoding: TokenEncoding) -> str:
    if isinstance(encoding, _LocalReversibleEncoding):
        return "local-reversible-v1"
    name = getattr(encoding, "name", None)
    return f"tiktoken:{name}" if isinstance(name, str) and name else "tiktoken:unknown"


def require_encoding(
    name: str = "cl100k_base",
    *,
    allow_fallback: bool = False,
) -> str:
    """Preflight the tokenizer used to create canonical chunk IDs."""
    encoding = get_encoding(name, allow_fallback=allow_fallback)
    backend = encoding_backend(encoding)
    if backend == "local-reversible-v1" and not (
        allow_fallback or _truthy_env("SCHOLAR_ALLOW_TOKENIZER_FALLBACK")
    ):
        raise TokenizerUnavailableError(f"Canonical tokenizer {name!r} is unavailable")
    return backend


def clear_encoding_cache() -> None:
    """Clear tokenizer resolution state (primarily for isolated tests)."""
    _get_encoding.cache_clear()


def count_tokens(text: str, *, encoding_name: str = "cl100k_base") -> int:
    if not text:
        return 0
    return len(get_encoding(encoding_name).encode(text))


def encode_tokens(text: str, *, encoding_name: str = "cl100k_base") -> list[int]:
    return list(get_encoding(encoding_name).encode(text))


def decode_tokens(tokens: list[int], *, encoding_name: str = "cl100k_base") -> str:
    return get_encoding(encoding_name).decode(tokens)

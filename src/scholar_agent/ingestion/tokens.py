"""Token counting helpers (real tokenizer counts, not character proxies)."""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=4)
def get_encoding(name: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def count_tokens(text: str, *, encoding_name: str = "cl100k_base") -> int:
    if not text:
        return 0
    return len(get_encoding(encoding_name).encode(text))


def encode_tokens(text: str, *, encoding_name: str = "cl100k_base") -> list[int]:
    return list(get_encoding(encoding_name).encode(text))


def decode_tokens(tokens: list[int], *, encoding_name: str = "cl100k_base") -> str:
    return get_encoding(encoding_name).decode(tokens)

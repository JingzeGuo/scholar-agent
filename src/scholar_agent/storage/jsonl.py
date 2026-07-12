"""Typed JSONL repository for canonical corpus artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class JsonlRepositoryError(ValueError):
    """Raised when JSONL read/write or schema validation fails."""


class JsonlRepository(Generic[T]):
    """Line-delimited JSON store backed by a Pydantic model type.

    Each non-empty line is one model instance. This is the persistence surface
    for papers, chunks, entities, and relations. The chunk store is the source
    of truth for retrieval indexes in later phases.
    """

    def __init__(self, path: Path | str, model_type: type[T]) -> None:
        self.path = Path(path)
        self.model_type = model_type

    def exists(self) -> bool:
        return self.path.is_file()

    def ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def iter_rows(self) -> Iterator[T]:
        if not self.exists():
            yield from ()
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise JsonlRepositoryError(
                        f"{self.path}:{line_no}: invalid JSON: {exc}"
                    ) from exc
                try:
                    yield self.model_type.model_validate(data)
                except ValidationError as exc:
                    raise JsonlRepositoryError(
                        f"{self.path}:{line_no}: schema validation failed for "
                        f"{self.model_type.__name__}: {exc}"
                    ) from exc

    def read_all(self) -> list[T]:
        return list(self.iter_rows())

    def write_all(self, items: Sequence[T], *, atomic: bool = True) -> None:
        self.ensure_parent()
        payload = "".join(item.model_dump_json() + "\n" for item in items)
        if not atomic:
            self.path.write_text(payload, encoding="utf-8")
            return
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self.path)

    def append(self, item: T) -> None:
        self.ensure_parent()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(item.model_dump_json() + "\n")

    def append_many(self, items: Iterable[T]) -> None:
        self.ensure_parent()
        with self.path.open("a", encoding="utf-8") as handle:
            for item in items:
                handle.write(item.model_dump_json() + "\n")

    def count(self) -> int:
        return sum(1 for _ in self.iter_rows())

    def index_by(self, key: str) -> dict[str, T]:
        """Build a dict keyed by a string field (e.g. ``chunk_id``)."""
        result: dict[str, T] = {}
        for item in self.iter_rows():
            value = getattr(item, key, None)
            if not isinstance(value, str):
                raise JsonlRepositoryError(
                    f"field {key!r} is not a string on {self.model_type.__name__}"
                )
            if value in result:
                raise JsonlRepositoryError(f"duplicate {key}={value!r} in {self.path}")
            result[value] = item
        return result

.PHONY: sync test lint quality ingest index ask demo

sync:
	UV_CACHE_DIR=/tmp/scholar-agent-uv-cache uv sync

test:
	UV_CACHE_DIR=/tmp/scholar-agent-uv-cache uv run pytest -q

lint:
	UV_CACHE_DIR=/tmp/scholar-agent-uv-cache uv run ruff check .

quality: lint test

ingest:
	UV_CACHE_DIR=/tmp/scholar-agent-uv-cache uv run scholar-agent ingest tests/fixtures/papers

index:
	UV_CACHE_DIR=/tmp/scholar-agent-uv-cache uv run scholar-agent index

ask:
	UV_CACHE_DIR=/tmp/scholar-agent-uv-cache uv run scholar-agent ask "Compare Self-RAG and CRAG"

demo:
	UV_CACHE_DIR=/tmp/scholar-agent-uv-cache uv run scholar-agent demo

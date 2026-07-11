.PHONY: sync install test lint typecheck quality compatibility prototype clean

sync:
	uv sync

install: sync

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-live:
	uv run pytest -m live

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src

quality: lint typecheck test

compatibility:
	uv run python scripts/deepseek_compatibility.py

prototype:
	uv run python -m scholar_agent.agents.prototype_loop

corpus-download:
	uv run python scripts/download_corpus.py --target 120 --skip-existing

corpus-validate:
	uv run scholar-agent corpus validate -m data/corpus_manifest.jsonl --check-pdfs

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

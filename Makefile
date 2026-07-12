.PHONY: sync install test lint typecheck quality compatibility prototype clean demo evaluate evaluate-smoke

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

ingest:
	uv run scholar-agent ingest --manifest data/corpus_manifest.jsonl

ingest-smoke:
	uv run scholar-agent ingest --manifest data/corpus_manifest.jsonl --limit 5

index-build:
	uv run scholar-agent index build --embedding-backend hash

index-build-st:
	uv run scholar-agent index build --embedding-backend st --force

retrieve-demo:
	uv run scholar-agent retrieve "What is Self-RAG?" --mode hybrid_rerank --embedding-backend hash --debug

ask-naive-demo:
	uv run scholar-agent ask-naive "What is Self-RAG?" --embedding-backend hash

graph-build:
	uv run scholar-agent graph build --force

graph-inspect:
	uv run scholar-agent graph inspect

research-demo:
	uv run scholar-agent research "Compare Self-RAG and CRAG" --embedding-backend hash

ask-demo:
	uv run scholar-agent ask "Compare Self-RAG versus CRAG" --max-iterations 2

evaluate-smoke:
	uv run scholar-agent evaluate \
		--system naive_dense --system hybrid_rerank \
		--max-questions 5 \
		--embedding-backend hash \
		--output-dir outputs/evaluation/smoke

evaluate:
	uv run scholar-agent evaluate \
		--eval-config configs/evaluation.yaml \
		--embedding-backend hash \
		--output-dir outputs/evaluation

demo:
	uv sync --extra ui
	uv run streamlit run src/scholar_agent/app/streamlit_app.py

demo-precompute:
	uv run python scripts/precompute_demo_runs.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

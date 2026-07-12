"""Load / save precomputed demo runs for offline interview replay."""

from __future__ import annotations

import json
from pathlib import Path

from scholar_agent.app.demo_models import SavedDemoRun
from scholar_agent.logging import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEMO_RUNS_DIR = REPO_ROOT / "data" / "demo" / "runs"


def demo_runs_dir(path: Path | str | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_DEMO_RUNS_DIR


def list_saved_runs(directory: Path | str | None = None) -> list[SavedDemoRun]:
    root = demo_runs_dir(directory)
    if not root.is_dir():
        return []
    runs: list[SavedDemoRun] = []
    for path in sorted(root.glob("*.json")):
        try:
            runs.append(load_saved_run(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip invalid demo run %s: %s", path, exc)
    return runs


def load_saved_run(path: Path | str) -> SavedDemoRun:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SavedDemoRun.model_validate(data)


def save_demo_run(
    run: SavedDemoRun,
    directory: Path | str | None = None,
    *,
    filename: str | None = None,
) -> Path:
    root = demo_runs_dir(directory)
    root.mkdir(parents=True, exist_ok=True)
    name = filename or f"{run.demo_id}.json"
    path = root / name
    path.write_text(
        json.dumps(run.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def find_saved_run(demo_id: str, directory: Path | str | None = None) -> SavedDemoRun | None:
    for run in list_saved_runs(directory):
        if run.demo_id == demo_id:
            return run
    return None

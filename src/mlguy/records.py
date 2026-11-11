"""Experiment records: the append-only history of submissions.

Only `mlguy validate` and `mlguy submit` write here. Each submission keeps a self-contained snapshot of
everything needed to reproduce its score (predict.py, src/, model/), so a later
experiment cannot change what an earlier one scored.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from mlguy import judge

SNAPSHOT_PATHS = ("predict.py", "src", "model")
SNAPSHOT_WARN_MB = 200


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_task(project_dir: Path) -> str:
    """A hash over every file in task/, so a changed task is noticed."""
    h = hashlib.sha256()
    task_dir = project_dir / "task"
    for path in sorted(p for p in task_dir.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(task_dir)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


class Experiments:
    """The experiments/ folder: log.jsonl plus one folder per submission."""

    def __init__(self, project_dir: Path):
        self.dir = Path(project_dir) / "experiments"
        self.log = self.dir / "log.jsonl"

    def create(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log.touch()

    def all(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def submissions(self) -> list[dict]:
        """Scored submissions only, baseline included; errors excluded."""
        return [r for r in self.all() if r["status"] == "ok"]

    def next_number(self) -> int:
        records = self.all()
        return max((r["number"] for r in records), default=-1) + 1

    def best(self, primary: str) -> dict | None:
        best = None
        for record in self.submissions():
            if best is None or judge.is_better(record["metrics"][primary], best["metrics"][primary], primary):
                best = record
        return best

    def append(self, record: dict) -> None:
        self.create()
        with self.log.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def replace(self, record: dict) -> None:
        """Rewrite the record with this number. Only the baseline, before any submission, is ever replaced."""
        kept = [r for r in self.all() if r["number"] != record["number"]]
        self.log.write_text("".join(json.dumps(r) + "\n" for r in sorted(kept + [record], key=lambda r: r["number"])))

    def predictions(self, number: int) -> Path:
        return self.dir / f"{number:04d}" / "validation_predictions.csv"

    def snapshot(self, project_dir: Path, number: int) -> tuple[Path, float]:
        """Copy the prediction side of the project into experiments/NNNN/submission/."""
        target = self.dir / f"{number:04d}" / "submission"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for name in SNAPSHOT_PATHS:
            source = Path(project_dir) / name
            if not source.exists():
                continue
            if source.is_dir():
                shutil.copytree(source, target / name, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(source, target / name)
        size_mb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e6
        return target, size_mb

    def write_detail(self, number: int, name: str, text: str) -> None:
        path = self.dir / f"{number:04d}" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def git_commit(project_dir: Path, message: str) -> str | None:
    """Commit the project if it is the root of a git work tree. Returns the commit id."""
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=project_dir, capture_output=True, text=True, timeout=30,
        )
        if top.returncode != 0 or Path(top.stdout.strip()) != Path(project_dir).resolve():
            return None
        subprocess.run(["git", "add", "-A"], cwd=project_dir, capture_output=True, timeout=60)
        subprocess.run(
            ["git", "commit", "-m", message, "--allow-empty"],
            cwd=project_dir, capture_output=True, text=True, timeout=60,
        )
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_dir, capture_output=True, text=True, timeout=30,
        )
        return head.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None

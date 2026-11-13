"""Experiment records: the append-only history of submissions.

Only `mlguy validate` and `mlguy submit` write here. Each submission keeps a self-contained snapshot of
everything needed to reproduce its score (infer.py, src/, model/), so a later
experiment cannot change what an earlier one scored.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SNAPSHOT_PATHS = ("infer.py", "src", "model")
SNAPSHOT_WARN_MB = 200


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tool_version() -> str:
    """What scored a submission. The task is frozen, but the tool that scores it is not: an
    upgrade mid-project can change what a score means, so every record says which one it was."""
    from mlguy import __version__

    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        if head.returncode == 0:
            changed = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
            return f"{__version__}+{head.stdout.strip()}" + ("-modified" if changed else "")
    except Exception:
        pass
    return __version__


SCORING_SOURCES = ("project.py",)


def scoring_digest() -> str:
    """A fingerprint of the framework code that turns a submission into a score.

    `tool_version` says which mlguy scored a submission, but most of what changes between two
    versions has nothing to do with scoring, and a warning that fires on every upgrade leaves
    the reader to diff the tool themselves. This narrows it to the files that decide what a
    score is, so an upgrade can be reported as harmless when it was.
    """
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in SCORING_SOURCES:
        path = here / name
        h.update(path.read_bytes() if path.exists() else b"")
    return h.hexdigest()[:12]


IGNORED_IN_TASK = ("__pycache__", ".ipynb_checkpoints", ".DS_Store")

# The task must not change once it is frozen, and that is checked on every submission. Proving
# it byte for byte would mean reading the whole dataset each time, so files are fingerprinted at
# a level of detail that scales: small ones by their contents, large ones by their size and
# their two ends, and a folder of very many files by the names and sizes it holds. This catches
# a task that was edited, regenerated or swapped; it is not meant to defeat a forgery.
SMALL_FILE = 4 << 20       # read a file this size or smaller in full
EDGE = 1 << 20             # of a larger file, read the first and last megabyte
CROWDED = 2000             # a folder holding more than this many files is read by its listing


def _task_files(task_dir: Path) -> list[Path]:
    return sorted(p for p in task_dir.rglob("*") if p.is_file() and p.suffix != ".pyc"
                  and not any(part in IGNORED_IN_TASK for part in p.relative_to(task_dir).parts))


def _fingerprint(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha256(f"{size}:".encode())
    with open(path, "rb") as f:
        if size <= SMALL_FILE:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        else:
            h.update(f.read(EDGE))
            f.seek(-EDGE, 2)
            h.update(f.read(EDGE))
    return h.hexdigest()


def _listing(folder: Path) -> str:
    """A folder of very many files, as the names and sizes it holds. No file is read."""
    files = _task_files(folder)
    h = hashlib.sha256()
    for path in files:
        h.update(f"{path.relative_to(folder)}:{path.stat().st_size}\n".encode())
    total = sum(p.stat().st_size for p in files)
    return f"{len(files)} files, {total} bytes, {h.hexdigest()}"


def task_manifest(project_dir: Path) -> dict[str, str]:
    """A fingerprint of everything in task/, by path, so a changed task is noticed and named.
    Caches that Python and editors write by themselves are ignored."""
    task_dir = Path(project_dir) / "task"
    manifest = {}
    for entry in sorted(task_dir.iterdir()) if task_dir.exists() else []:
        if entry.name in IGNORED_IN_TASK:
            continue
        if entry.is_dir() and len(_task_files(entry)) > CROWDED:
            manifest[f"{entry.name}/"] = _listing(entry)
            continue
        for path in ([entry] if entry.is_file() else _task_files(entry)):
            if path.suffix != ".pyc":
                manifest[str(path.relative_to(task_dir))] = _fingerprint(path)
    return manifest


def hash_task(project_dir: Path) -> str:
    """One hash over the whole of task/, for display."""
    return hashlib.sha256(json.dumps(task_manifest(project_dir), sort_keys=True).encode()).hexdigest()


def is_better(new: float, old: float | None, direction: str) -> bool:
    if old is None:
        return True
    return new > old if direction == "maximize" else new < old


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

    def best(self, primary: str, direction: str) -> dict | None:
        best = None
        for record in self.submissions():
            if best is None or is_better(record["metrics"][primary], best["metrics"][primary], direction):
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

    def outputs(self, number: int) -> Path:
        """What the submission produced for the validation examples, exactly as written."""
        return self.dir / f"{number:04d}" / "validation_outputs"

    @staticmethod
    def megabytes(path: Path) -> float | None:
        """Size of a file or folder on disk, or None if it is not there."""
        path = Path(path)
        if not path.exists():
            return None
        if path.is_file():
            return path.stat().st_size / 1e6
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6

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
        return target, self.megabytes(target) or 0.0

    def write_detail(self, number: int, name: str, text: str) -> None:
        path = self.dir / f"{number:04d}" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def git_commit(project_dir: Path, message: str) -> str | None:
    """Commit the project if it is the root of a git work tree. Returns the commit id.

    A failure is said out loud rather than swallowed: the files are still in place, but a history
    that silently stops at the last submission reads as though `finish` never ran."""
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=project_dir, capture_output=True, text=True, timeout=30,
        )
        if top.returncode != 0 or Path(top.stdout.strip()) != Path(project_dir).resolve():
            return None
        for step in (["git", "add", "-A"], ["git", "commit", "-q", "-m", message, "--allow-empty"]):
            done = subprocess.run(step, cwd=project_dir, capture_output=True, text=True, timeout=300)
            if done.returncode != 0:
                said = (done.stderr or done.stdout).strip().splitlines()
                raise subprocess.SubprocessError(said[-1] if said else f"{' '.join(step[:2])} failed")
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_dir, capture_output=True, text=True, timeout=30,
        )
        return head.stdout.strip() or None
    except (OSError, subprocess.SubprocessError) as e:
        print(f"warning: could not commit to the project's git repository ({e}). The files are in "
              "place; commit them yourself. A large file the .gitignore does not exclude is the usual "
              "cause.", file=sys.stderr)
        return None

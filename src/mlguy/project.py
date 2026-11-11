"""A project on disk: the frozen task, where hidden test labels live, and running its predict.py."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pandas as pd

from mlguy import judge
from mlguy.judge import SubmissionError

PREDICT_TIMEOUT_S = 3600


class TaskChanged(Exception):
    """task/ differs from what was approved, so scores would not be comparable."""


def load_task(project_dir: str | Path) -> dict:
    project_dir = Path(project_dir).resolve()
    path = project_dir / "task" / "task.toml"
    if not path.exists():
        raise FileNotFoundError(f"no task at {path}; is {project_dir} a mlguy project?")
    with open(path, "rb") as f:
        task = tomllib.load(f)
    task["_project_dir"] = project_dir
    return task


def state_dir(task: dict) -> Path:
    """Where things the project's code must not see are kept: outside the project folder."""
    home = Path(os.environ.get("MLGUY_HOME", Path.home() / ".mlguy"))
    return home / "projects" / task["name"]


def test_labels_path(task: dict) -> Path:
    return state_dir(task) / "test_labels.csv"


def task_hash_path(task: dict) -> Path:
    return state_dir(task) / "task_hash.txt"


def store_test_labels(task: dict, labels: pd.DataFrame) -> Path:
    path = test_labels_path(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(path, index=False)
    return path


def check_task_unchanged(task: dict) -> None:
    """Raise if task/ no longer matches what was approved at init."""
    from mlguy.records import hash_task

    recorded = task_hash_path(task)
    if not recorded.exists():
        raise TaskChanged(f"this project was never initialised on this machine; run `mlguy init` first")
    if hash_task(task["_project_dir"]) != recorded.read_text().strip():
        raise TaskChanged(
            "task/ has changed since the brief was approved. The metric, the data and the split are "
            "fixed; restore them, or start a new project if the task itself should change."
        )


def run_predict(run_dir: Path, input_csv: Path, output_csv: Path, timeout: int = PREDICT_TIMEOUT_S) -> None:
    """Call the one fixed interface: predict.py --model DIR --input FILE --output FILE."""
    if not (run_dir / "predict.py").exists():
        raise SubmissionError(f"no predict.py in {run_dir}")
    if not (run_dir / "model").exists():
        raise SubmissionError(f"no model/ directory in {run_dir}; train a model before scoring")
    cmd = [sys.executable, "predict.py", "--model", "model", "--input", str(input_csv), "--output", str(output_csv)]
    try:
        proc = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SubmissionError(f"predict.py did not finish within {timeout} s")
    if proc.returncode != 0:
        raise SubmissionError(f"predict.py failed (exit {proc.returncode}):\n{proc.stderr.strip()[-2000:]}")
    if not output_csv.exists():
        raise SubmissionError("predict.py exited cleanly but wrote no output file")


def score_split(project_dir: str | Path, split: str, run_dir: str | Path | None = None) -> dict:
    """Run predict.py on one split's inputs (labels withheld) and score the result.

    `run_dir` is where predict.py, src/ and model/ live; it defaults to the project itself,
    and is an experiment snapshot when scoring a past submission.
    """
    task = load_task(project_dir)
    project_dir = task["_project_dir"]
    run_dir = Path(run_dir) if run_dir else project_dir
    data, seq_col, tgt_col = task["data"], task["data"]["sequence_column"], task["data"]["target_column"]

    if split == "validation":
        df = pd.read_csv(project_dir / "task" / data["validation"])
        inputs, y_true = df[[seq_col]], df[tgt_col].to_numpy()
    elif split == "test":
        labels_file = test_labels_path(task)
        if not labels_file.exists():
            raise FileNotFoundError(f"no test labels at {labels_file}; run `mlguy init` on this machine")
        inputs = pd.read_csv(project_dir / "task" / data["test_inputs"])[[seq_col]]
        y_true = pd.read_csv(labels_file)[tgt_col].to_numpy()
        if len(inputs) != len(y_true):
            raise SubmissionError(f"{len(inputs)} test inputs but {len(y_true)} stored labels")
    else:
        raise ValueError("split must be 'validation' or 'test'")

    with tempfile.TemporaryDirectory() as tmp:
        input_csv, output_csv = Path(tmp) / "inputs.csv", Path(tmp) / "predictions.csv"
        inputs.to_csv(input_csv, index=False)
        run_predict(run_dir, input_csv, output_csv)
        try:
            preds = pd.read_csv(output_csv)
        except Exception as e:
            raise SubmissionError(f"could not read the predictions file: {e}")

    y_pred = judge.check_predictions(preds, len(y_true))
    primary = task["metric"]["primary"]
    names = [primary] + [m for m in task["metric"].get("secondary", []) if m != primary]
    return {"split": split, "n": int(len(y_true)), "primary": primary, "metrics": judge.score(y_true, y_pred, names)}

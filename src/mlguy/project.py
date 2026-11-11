"""A project on disk: the frozen task, what is held back, and the two steps of scoring.

Scoring is always the same two steps, whatever the problem:

  1. produce   run the project's entry point to turn inputs into outputs
               prediction:  predict.py  --model DIR --input FILE --output FILE
               generation:  generate.py --model DIR [--conditions FILE] --n N --seed S --output FILE
  2. evaluate  turn those outputs into numbers
               by default:  a built-in metric against held-back labels
               otherwise:   the task's own evaluator, task/<script>, frozen with the task
"""

from __future__ import annotations

import json
import math
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
EVALUATOR_TIMEOUT_MIN = 60


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


def validation_caveat(task: dict) -> str | None:
    """The warning to carry everywhere when validation does not ask the test set's question."""
    data = task.get("data", {})
    if data.get("validation_matches_test", True):
        return None
    note = data.get("validation_note", "").strip()
    return (
        "Validation does not ask the same question as the test set, so a better validation "
        "score does not reliably mean a better model." + (f" {note}" if note else "")
    )


def state_dir(task: dict) -> Path:
    """Where things the project's code must not see are kept: outside the project folder."""
    home = Path(os.environ.get("MLGUY_HOME", Path.home() / ".mlguy"))
    return home / "projects" / task["name"]


def test_labels_path(task: dict) -> Path:
    """The held-back test rows: labels for prediction, conditions and references for generation."""
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


def input_columns(task: dict) -> list[str]:
    """The columns handed to the model: what it predicts from, or the conditions it generates
    for. Anything: text, numbers, categories, file paths. Empty means unconditional generation."""
    return list(task["data"].get("input_columns", []))


def is_generation(task: dict) -> bool:
    return task["task"]["type"] == "generation"


def entry_point(task: dict) -> str:
    return "generate.py" if is_generation(task) else "predict.py"


def evaluator_script(task: dict) -> Path | None:
    """The task's own evaluator, or None when a built-in metric against labels is used."""
    script = task.get("evaluator", {}).get("script")
    return task["_project_dir"] / "task" / script if script else None


def direction(task: dict) -> str:
    """'maximize' or 'minimize' for the primary metric."""
    if evaluator_script(task) is None:
        return judge.METRICS[task["metric"]["primary"]][1]
    return task["metric"]["direction"]


def sampling(task: dict, split: str) -> tuple[int, int]:
    """(samples per condition, seed). Fixed by the task so two submissions are comparable; the
    final check draws fresh samples, and more of them if the task says so."""
    spec = task.get("sampling", {})
    n, seed = int(spec.get("n", 100)), int(spec.get("seed", 0))
    if split == "test":
        return int(spec.get("final_n", n)), int(spec.get("final_seed", seed + 1))
    return n, seed


def _run_entry(run_dir: Path, script: str, arguments: list[str], output: Path, timeout: int) -> None:
    if not (run_dir / script).exists():
        raise SubmissionError(f"no {script} in {run_dir}")
    if not (run_dir / "model").exists():
        raise SubmissionError(f"no model/ directory in {run_dir}; train a model before scoring")
    cmd = [sys.executable, script, "--model", "model", *arguments, "--output", str(output)]
    try:
        proc = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SubmissionError(f"{script} did not finish within {timeout} s")
    if proc.returncode != 0:
        raise SubmissionError(f"{script} failed (exit {proc.returncode}):\n{proc.stderr.strip()[-2000:]}")
    if not output.exists():
        raise SubmissionError(f"{script} exited cleanly but wrote no output file")


def run_predict(run_dir: Path, input_csv: Path, output_csv: Path, timeout: int = PREDICT_TIMEOUT_S) -> None:
    """predict.py --model DIR --input FILE --output FILE: one output per input row."""
    _run_entry(run_dir, "predict.py", ["--input", str(input_csv)], output_csv, timeout)


def run_generate(run_dir: Path, conditions_csv: Path | None, n: int, seed: int, output_csv: Path,
                 timeout: int = PREDICT_TIMEOUT_S) -> None:
    """generate.py --model DIR [--conditions FILE] --n N --seed S --output FILE: n samples per condition."""
    arguments = (["--conditions", str(conditions_csv)] if conditions_csv else []) + ["--n", str(n), "--seed", str(seed)]
    _run_entry(run_dir, "generate.py", arguments, output_csv, timeout)


def _reference(task: dict, split: str) -> Path | None:
    """The rows of this split that the evaluator may see and the model may not."""
    if split == "validation":
        name = task["data"].get("validation")
        return task["_project_dir"] / "task" / name if name else None
    if split == "test":
        path = test_labels_path(task)
        if path.exists():
            return path
        if is_generation(task):
            return _reference(task, "validation")  # no held-out conditions: fresh samples of the same ones
        raise FileNotFoundError(f"no test labels at {path}; run `mlguy init` on this machine")
    raise ValueError("split must be 'validation' or 'test'")


def _produce(task: dict, split: str, run_dir: Path, reference: Path | None, output: Path) -> None:
    columns, work = input_columns(task), output.parent
    if not is_generation(task):
        source = reference if split == "validation" else task["_project_dir"] / "task" / task["data"].get("test_inputs", "data/test_inputs.csv")
        pd.read_csv(source)[columns].to_csv(work / "inputs.csv", index=False)
        run_predict(run_dir, work / "inputs.csv", output)
        return
    n, seed = sampling(task, split)
    conditions = None
    if columns:
        conditions = pd.read_csv(reference)[columns]
        conditions.to_csv(work / "conditions.csv", index=False)
    run_generate(run_dir, work / "conditions.csv" if columns else None, n, seed, output)
    judge.check_generated(_read_outputs(output), n, None if conditions is None else len(conditions))


def _read_outputs(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise SubmissionError(f"could not read the output file: {e}")


def _builtin_scores(task: dict, reference: Path, outputs: Path, keep: Path | None) -> tuple[dict, list | None, int]:
    y_true = pd.read_csv(reference)[task["data"]["target_column"]].to_numpy()
    y_pred, y_score = judge.check_predictions(_read_outputs(outputs), len(y_true))
    primary = task["metric"]["primary"]
    names = [primary] + [m for m in task["metric"].get("secondary", []) if m != primary]
    metrics = judge.score(y_true, y_pred, names, y_score=y_score)
    if keep is not None:
        saved = pd.DataFrame({"prediction": y_pred})
        if y_score is not None:
            saved["score"] = y_score
        saved.to_csv(keep, index=False)
    return metrics, judge.interval(y_true, y_pred, primary, y_score=y_score), int(len(y_true))


def _evaluator_scores(task: dict, split: str, reference: Path | None, outputs: Path,
                      keep: Path | None) -> tuple[dict, list | None, int]:
    """Run the task's evaluator:

        python <script> --outputs FILE --split validation|final --result FILE [--reference FILE]

    It writes JSON: {"metrics": {name: number, ...}, "per_item": [number, ...]}. `per_item` is
    optional: one number per scored item whose mean is the primary metric. With it the score
    gets an interval and submissions can be compared; without it, neither.
    """
    script, primary = evaluator_script(task), task["metric"]["primary"]
    result_file = outputs.parent / "result.json"
    cmd = [sys.executable, str(script), "--outputs", str(outputs),
           "--split", "final" if split == "test" else "validation", "--result", str(result_file)]
    if reference is not None:
        cmd += ["--reference", str(reference)]
    timeout = 60 * int(task["evaluator"].get("timeout_minutes", EVALUATOR_TIMEOUT_MIN))
    try:
        proc = subprocess.run(cmd, cwd=script.parent, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SubmissionError(f"the evaluator did not finish within {timeout // 60} minutes")
    if proc.returncode != 0:
        raise SubmissionError(f"the evaluator could not score these outputs (exit {proc.returncode}):\n"
                              f"{proc.stderr.strip()[-2000:]}")
    try:
        result = json.loads(result_file.read_text())
        metrics = {k: float(v) for k, v in result["metrics"].items()}
    except Exception as e:
        raise SubmissionError(f"the evaluator did not write a usable result file: {e}")
    if primary not in metrics or not all(math.isfinite(v) for v in metrics.values()):
        raise SubmissionError(f"the evaluator must report a finite '{primary}'; it reported {metrics}")
    metrics = {primary: metrics[primary], **{k: v for k, v in metrics.items() if k != primary}}

    items = result.get("per_item")
    if items is not None and keep is not None:
        pd.DataFrame({"item": items}).to_csv(keep, index=False)
    return metrics, judge.mean_interval(items) if items else None, int(len(_read_outputs(outputs)))


def score_split(project_dir: str | Path, split: str, run_dir: str | Path | None = None,
                keep_predictions: Path | None = None, outputs: str | Path | None = None) -> dict:
    """Produce outputs for one split and evaluate them.

    `run_dir` is where the entry point, src/ and model/ live; it defaults to the project itself,
    and is an experiment snapshot when scoring a past submission. `keep_predictions` saves what
    is needed to compare a later submission with this one. `outputs` skips the first step and
    evaluates a file that already exists.
    """
    task = load_task(project_dir)
    run_dir = Path(run_dir) if run_dir else task["_project_dir"]
    reference = _reference(task, split)
    if keep_predictions is not None:
        keep_predictions.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        produced = Path(tmp) / "outputs.csv"
        if outputs is not None:
            produced = Path(outputs).resolve()
            if not produced.exists():
                raise SubmissionError(f"no file at {produced}")
        else:
            _produce(task, split, run_dir, reference, produced)
        if evaluator_script(task) is not None:
            scratch = Path(tmp) / "evaluation"
            scratch.mkdir()
            staged = scratch / "outputs.csv"
            staged.symlink_to(produced)
            metrics, interval, n = _evaluator_scores(task, split, reference, staged, keep_predictions)
        else:
            metrics, interval, n = _builtin_scores(task, reference, produced, keep_predictions)

    return {"split": split, "n": n, "primary": task["metric"]["primary"], "metrics": metrics, "interval": interval}


def compare_with(project_dir: Path, new_file: Path, old_file: Path) -> float | None:
    """How often the newer validation outputs beat the older ones, over resampled rows."""
    task = load_task(project_dir)
    if not (new_file.exists() and old_file.exists()):
        return None
    new, old = pd.read_csv(new_file), pd.read_csv(old_file)
    if len(new) != len(old):
        return None
    if evaluator_script(task) is not None:
        return judge.mean_chance_better(new["item"].to_numpy(dtype=float), old["item"].to_numpy(dtype=float),
                                        direction(task))
    y_true = pd.read_csv(_reference(task, "validation"))[task["data"]["target_column"]].to_numpy(dtype=float)
    if len(new) != len(y_true):
        return None
    pairs = [(f["prediction"].to_numpy(dtype=float), f["score"].to_numpy(dtype=float) if "score" in f else None)
             for f in (new, old)]
    return judge.chance_better(y_true, pairs[0], pairs[1], task["metric"]["primary"])

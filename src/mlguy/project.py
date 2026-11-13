"""A project on disk: the frozen task, what is held back, and the two steps of scoring.

Scoring is always the same two steps, whatever the problem:

  1. produce   run the project's one entry point to turn a split's inputs into outputs
                   infer.py --model DIR [--input PATH] --output PATH [any [infer] args]
  2. evaluate  turn those outputs into numbers with the task's own evaluator, task/evaluate.py
                   evaluate.py --outputs PATH [--reference PATH] [--inputs PATH]
                               --split validation|final --result FILE

The framework knows no metrics, no formats and no kinds of task. A split is up to two things:
inputs, which the model is given, and a reference, which only the evaluator sees. Either may be
a table, a folder, anything, or absent. What the framework owns is which of them each step is
given and when, that the test split stays out of the project until the final check, and that
the scoring cannot change.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path


INFER_TIMEOUT_S = 3600
EVALUATOR_TIMEOUT_MIN = 60


class TaskChanged(Exception):
    """task/ differs from what was approved, so scores would not be comparable."""


class SubmissionError(Exception):
    """The submission could not be scored (crash, bad output, no usable score). Never a zero."""


class EvaluatorFailed(SubmissionError):
    """The model ran, and the evaluator would not score what it produced. The evaluator had the
    split's reference when it decided, so on the test split this is the final check's result."""


# Keys of the first contract, which knew two kinds of task. Ignoring them would be worse than
# refusing: a test_labels file nobody moves out of the project is a test set the model can read.
_OLD_KEYS = ("validation_labels", "test_labels", "id_column", "target_column", "validation_matches_test")


def load_task(project_dir: str | Path) -> dict:
    project_dir = Path(project_dir).resolve()
    path = project_dir / "task" / "task.toml"
    if not path.exists():
        raise FileNotFoundError(f"no task at {path}; is {project_dir} a mlguy project?")
    with open(path, "rb") as f:
        task = tomllib.load(f)
    old = [k for k in _OLD_KEYS if k in task.get("data", {})] + (["sampling"] if "sampling" in task else [])
    if old:
        raise FileNotFoundError(
            f"{path} uses keys from an older mlguy: {', '.join(old)}. Rename validation_labels and "
            "test_labels to validation_reference and test_reference, move [sampling] n and seed to "
            "[infer] args = {n = .., seed = ..} and final_n and final_seed to [infer] final_args, "
            "and drop id_column, target_column and validation_matches_test.")
    task["_project_dir"] = project_dir
    return task


def validation_caveat(task: dict) -> str | None:
    """What the user was told about the split, to carry into the report word for word."""
    return task.get("data", {}).get("validation_note", "").strip() or None


STATE_POINTER = ".mlguy-home"


def mlguy_home(project_dir: Path) -> Path:
    """MLGUY_HOME if set; otherwise wherever this project was initialised; otherwise ~/.mlguy.
    `mlguy init` records the location in the project, so it need not be set on every command."""
    if os.environ.get("MLGUY_HOME"):
        return Path(os.environ["MLGUY_HOME"]).resolve()
    pointer = Path(project_dir) / STATE_POINTER
    if pointer.exists():
        return Path(pointer.read_text().strip())
    return Path.home() / ".mlguy"


def state_dir(task: dict) -> Path:
    """Where things the project's code must not see are kept: outside the project folder."""
    return mlguy_home(task["_project_dir"]) / "projects" / task["name"]


TEST_KEYS = ("test_inputs", "test_reference")


def held_back_path(task: dict, key: str) -> Path:
    """Where `init` keeps a declared part of the test split, outside the project."""
    return state_dir(task) / (key + Path(task["data"][key]).suffix)


def has_test_split(task: dict) -> bool:
    """Whether the task holds anything back. If not, the final check runs the model again on
    validation, with the final arguments: fresh samples of the same conditions, say."""
    return any(key in task.get("data", {}) for key in TEST_KEYS)


def task_changes(task: dict) -> list[str]:
    """The files in task/ that differ from what was frozen at init, or [] if none do."""
    from mlguy.records import task_manifest

    recorded = state_dir(task) / "task_files.json"
    if not recorded.exists():
        raise TaskChanged("this project was never initialised on this machine; run `mlguy init` first")
    before, now = json.loads(recorded.read_text()), task_manifest(task["_project_dir"])
    changed = sorted(name for name in before.keys() & now.keys() if before[name] != now[name])
    return ([f"{n} (changed)" for n in changed] + [f"{n} (removed)" for n in sorted(before.keys() - now.keys())]
            + [f"{n} (added)" for n in sorted(now.keys() - before.keys())])


def check_task_unchanged(task: dict) -> None:
    """Raise if task/ no longer matches what was approved at init."""
    changes = task_changes(task)
    if changes:
        raise TaskChanged(
            "task/ has changed since the brief was approved: " + ", ".join(changes[:5])
            + (f" and {len(changes) - 5} more" if len(changes) > 5 else "") + ". The metric, the data "
            "and the split are fixed; restore them, or start a new project if the task itself should change."
        )


ENTRY_POINT = "infer.py"


def data_path(task: dict, key: str) -> Path | None:
    name = task.get("data", {}).get(key)
    return task["_project_dir"] / "task" / name if name else None


def evaluator_script(task: dict) -> Path:
    """The task's evaluator: every task carries its own, frozen with the rest of task/."""
    return task["_project_dir"] / "task" / task.get("evaluator", {}).get("script", "evaluate.py")


def direction(task: dict) -> str:
    """'maximize' or 'minimize' for the primary metric."""
    return task["metric"]["direction"]


def infer_args(task: dict, split: str) -> list[str]:
    """Extra arguments for infer.py, fixed by the task so that submissions are comparable:
    `[infer] args = {n = 20, seed = 0}` becomes `--n 20 --seed 0`. The final check uses
    `final_args` when the task gives them (more samples, a fresh seed), and `args` otherwise."""
    spec = task.get("infer", {})
    chosen = spec.get("final_args", spec.get("args", {})) if split == "test" else spec.get("args", {})
    return [part for key, value in chosen.items() for part in (f"--{key}", str(value))]


def run_infer(run_dir: Path, inputs: Path | None, output: Path, extra: list[str],
              timeout: int = INFER_TIMEOUT_S) -> None:
    """The one fixed interface: infer.py --model DIR [--input PATH] --output PATH [extra]."""
    if not (run_dir / ENTRY_POINT).exists():
        raise SubmissionError(f"no {ENTRY_POINT} in {run_dir}")
    if not (run_dir / "model").exists():
        raise SubmissionError(f"no model/ directory in {run_dir}; train a model before scoring")
    cmd = [sys.executable, ENTRY_POINT, "--model", "model"]
    if inputs is not None:
        cmd += ["--input", str(inputs)]
    cmd += ["--output", str(output), *extra]
    try:
        proc = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SubmissionError(f"{ENTRY_POINT} did not finish within {timeout} s")
    if proc.returncode != 0:
        raise SubmissionError(f"{ENTRY_POINT} failed (exit {proc.returncode}):\n{proc.stderr.strip()[-2000:]}")
    if not output.exists():
        raise SubmissionError(f"{ENTRY_POINT} exited cleanly but wrote nothing to {output.name}")


def _held_back_gone(task: dict, key: str, path: Path) -> FileNotFoundError:
    """Scoring validation instead would report a validation number as the final one."""
    return FileNotFoundError(
        f"task.toml declares data.{key}, so it was moved out of the project at `mlguy init` and "
        f"kept at {path}. It is not there now. The final check cannot be scored against a test set "
        f"that is missing, and scoring it against validation instead would report a validation "
        f"number as the final one. Restore {path.parent}, or point MLGUY_HOME at wherever this "
        f"project's state was kept."
    )


def split_part(task: dict, split: str, part: str) -> Path | None:
    """A split's `inputs` or `reference`, wherever it is kept now; None when the task has none."""
    if split not in ("validation", "test"):
        raise ValueError("split must be 'validation' or 'test'")
    if split == "validation" or not has_test_split(task):
        return data_path(task, f"validation_{part}")
    key = f"test_{part}"
    if key not in task["data"]:
        return None
    if not held_back_path(task, key).exists():
        raise _held_back_gone(task, key, held_back_path(task, key))
    return held_back_path(task, key)


def _keep(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)


def _evaluator_scores(task: dict, split: str, outputs: Path) -> dict:
    """One call of the task's evaluator. It writes {"metrics": {name: number, ...}} with the
    primary metric included and finite. Returns the metrics, primary first."""
    script, primary = evaluator_script(task), task["metric"]["primary"]
    result_file = outputs.parent / "result.json"
    cmd = [sys.executable, str(script), "--outputs", str(outputs),
           "--split", "final" if split == "test" else "validation", "--result", str(result_file)]
    for part in ("reference", "inputs"):
        path = split_part(task, split, part)
        if path is not None:
            cmd += [f"--{part}", str(path)]
    timeout = 60 * int(task.get("evaluator", {}).get("timeout_minutes", EVALUATOR_TIMEOUT_MIN))
    try:
        proc = subprocess.run(cmd, cwd=script.parent, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise EvaluatorFailed(f"the evaluator did not finish within {timeout // 60} minutes")
    if proc.returncode != 0:
        raise EvaluatorFailed(f"the evaluator could not score these outputs (exit {proc.returncode}):\n"
                              f"{proc.stderr.strip()[-2000:]}")
    try:
        metrics = {k: float(v) for k, v in json.loads(result_file.read_text())["metrics"].items()}
    except Exception as e:
        raise EvaluatorFailed(f"the evaluator did not write a usable result file: {e}")
    if primary not in metrics:
        raise EvaluatorFailed(f"the evaluator must report '{primary}'; it reported {sorted(metrics)}")
    if not math.isfinite(metrics[primary]):
        raise EvaluatorFailed(f"the evaluator reported {primary} = {metrics[primary]}; it must be a finite number")
    return {primary: metrics[primary], **{k: v for k, v in metrics.items() if k != primary}}


def score_split(project_dir: str | Path, split: str, run_dir: str | Path | None = None,
                outputs: str | Path | None = None, keep_outputs: Path | None = None) -> dict:
    """Produce outputs for one split and evaluate them: one run, one score.

    `run_dir` is where infer.py, src/ and model/ live; it defaults to the project itself,
    and is an experiment snapshot when scoring a past submission. `outputs` skips the first
    step and evaluates outputs that already exist. `keep_outputs` saves what the model produced.
    """
    task = load_task(project_dir)
    run_dir = Path(run_dir) if run_dir else task["_project_dir"]

    with tempfile.TemporaryDirectory() as tmp:
        produced = Path(tmp) / "outputs"
        predict_seconds = None
        if outputs is not None:
            produced = Path(outputs).resolve()
            if not produced.exists():
                raise SubmissionError(f"nothing at {produced}")
        else:
            # How long the model takes to run is a fact about the submission that cannot be
            # recovered afterwards, and it is one of the things that decides whether a model is
            # worth its score. Measured here because this is the one place that runs infer.py.
            started = time.perf_counter()
            run_infer(run_dir, split_part(task, split, "inputs"), produced, infer_args(task, split))
            predict_seconds = time.perf_counter() - started
        if keep_outputs is not None:
            _keep(produced, keep_outputs)
        scratch = Path(tmp) / "evaluation"
        scratch.mkdir()
        staged = scratch / "outputs"
        staged.symlink_to(produced)
        metrics = _evaluator_scores(task, split, staged)

    return {"split": split, "primary": task["metric"]["primary"], "metrics": metrics,
            "predict_seconds": predict_seconds}

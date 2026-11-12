"""A project on disk: the frozen task, what is held back, and the two steps of scoring.

Scoring is always the same two steps, whatever the problem:

  1. produce   run the project's one entry point to turn inputs into outputs
                   infer.py --model DIR --input PATH --output FILE
               (generation, experimental, adds --n N --seed S)
  2. evaluate  turn those outputs into numbers
               by default:  a built-in metric against held-back labels
               otherwise:   the task's own evaluator, task/<script>, frozen with the task

The inputs are opaque. They can be a table, a matrix, a folder of images or structures: only
the project's own code reads them. The framework knows two things about an example, its id
and (for a built-in metric) its label, and matches outputs to labels by id.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pandas as pd

from mlguy import judge
from mlguy.judge import SubmissionError

INFER_TIMEOUT_S = 3600
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
    """The caveat to carry everywhere: a validation/test mismatch, and any note on the split."""
    data = task.get("data", {})
    note = data.get("validation_note", "").strip()
    if data.get("validation_matches_test", True):
        return note or None
    return (
        "Validation does not ask the same question as the test set, so a better validation "
        "score does not reliably mean a better model." + (f" {note}" if note else "")
    )


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


def test_labels_path(task: dict) -> Path:
    """The held-back half of the test set: an id per example, and whatever it is scored against."""
    return state_dir(task) / "test_labels.csv"


def hidden_inputs_path(task: dict) -> Path:
    """Generation only: test conditions are kept out of the project until the final check."""
    return state_dir(task) / ("test_inputs" + Path(task["data"]["test_inputs"]).suffix)


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


def id_column(task: dict) -> str:
    return task["data"].get("id_column", "id")


def target_column(task: dict) -> str:
    return task["data"].get("target_column", "target")


def data_path(task: dict, key: str) -> Path | None:
    name = task["data"].get(key)
    return task["_project_dir"] / "task" / name if name else None


def is_generation(task: dict) -> bool:
    return task["task"]["type"] == "generation"


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
    """Generation: (samples per condition, seed). Fixed by the task so two submissions are
    comparable; the final check draws fresh samples, and more of them if the task says so."""
    spec = task.get("sampling", {})
    n, seed = int(spec.get("n", 100)), int(spec.get("seed", 0))
    if split == "test":
        return int(spec.get("final_n", n)), int(spec.get("final_seed", seed + 1))
    return n, seed


def run_infer(run_dir: Path, inputs: Path | None, output: Path, n: int | None = None, seed: int | None = None,
              timeout: int = INFER_TIMEOUT_S) -> None:
    """The one fixed interface: infer.py --model DIR --input PATH --output FILE."""
    if not (run_dir / ENTRY_POINT).exists():
        raise SubmissionError(f"no {ENTRY_POINT} in {run_dir}")
    if not (run_dir / "model").exists():
        raise SubmissionError(f"no model/ directory in {run_dir}; train a model before scoring")
    cmd = [sys.executable, ENTRY_POINT, "--model", "model"]
    if inputs is not None:
        cmd += ["--input", str(inputs)]
    if n is not None:
        cmd += ["--n", str(n), "--seed", str(seed)]
    cmd += ["--output", str(output)]
    try:
        proc = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SubmissionError(f"{ENTRY_POINT} did not finish within {timeout} s")
    if proc.returncode != 0:
        raise SubmissionError(f"{ENTRY_POINT} failed (exit {proc.returncode}):\n{proc.stderr.strip()[-2000:]}")
    if not output.exists():
        raise SubmissionError(f"{ENTRY_POINT} exited cleanly but wrote no output file")


def _reference(task: dict, split: str) -> Path | None:
    """The half of a split that the evaluator sees and the model does not: ids, and labels."""
    if split == "validation":
        return data_path(task, "validation_labels")
    if split == "test":
        path = test_labels_path(task)
        if path.exists():
            return path
        if is_generation(task):
            return _reference(task, "validation")  # no held-out conditions: fresh samples of the same ones
        raise FileNotFoundError(f"no test labels at {path}; run `mlguy init` on this machine")
    raise ValueError("split must be 'validation' or 'test'")


def _inputs(task: dict, split: str) -> Path | None:
    """The half of a split handed to the model, exactly as the task stores it."""
    if split == "test" and not is_generation(task):
        return data_path(task, "test_inputs")
    if split == "test" and "test_inputs" in task["data"] and hidden_inputs_path(task).exists():
        return hidden_inputs_path(task)
    return data_path(task, "validation_inputs")


def read_ids(task: dict, reference: Path) -> list[str]:
    return pd.read_csv(reference, dtype={id_column(task): str})[id_column(task)].tolist()


def _read_outputs(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise SubmissionError(f"could not read the output file: {e}")


def _produce(task: dict, split: str, run_dir: Path, reference: Path | None, output: Path) -> None:
    if not is_generation(task):
        run_infer(run_dir, _inputs(task, split), output)
        return
    n, seed = sampling(task, split)
    run_infer(run_dir, _inputs(task, split), output, n=n, seed=seed)
    judge.check_generated(_read_outputs(output), n, read_ids(task, reference) if reference else None)


def _builtin_scores(task: dict, reference: Path, outputs: Path) -> tuple[dict, int]:
    labels = pd.read_csv(reference, dtype={id_column(task): str})
    y_true = labels[target_column(task)].to_numpy()
    y_pred, y_score = judge.check_predictions(_read_outputs(outputs), labels[id_column(task)].tolist(),
                                              numeric=task["task"]["type"] != "classification")
    primary = task["metric"]["primary"]
    names = [primary] + [m for m in task["metric"].get("secondary", []) if m != primary]
    return judge.score(y_true, y_pred, names, y_score=y_score), int(len(y_true))


def _evaluator_scores(task: dict, split: str, reference: Path | None, outputs: Path) -> tuple[dict, int]:
    """One call of the task's evaluator:

        python <script> --outputs FILE --reference LABELS --split validation|final --result FILE

    It writes JSON {"metrics": {name: number, ...}} with the primary metric included and finite.
    Returns the metrics, primary first.
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
        metrics = {k: float(v) for k, v in json.loads(result_file.read_text())["metrics"].items()}
    except Exception as e:
        raise SubmissionError(f"the evaluator did not write a usable result file: {e}")
    if primary not in metrics:
        raise SubmissionError(f"the evaluator must report '{primary}'; it reported {sorted(metrics)}")
    if not math.isfinite(metrics[primary]):
        raise SubmissionError(f"the evaluator reported {primary} = {metrics[primary]}; it must be a finite number")
    return {primary: metrics[primary], **{k: v for k, v in metrics.items() if k != primary}}, \
        int(len(_read_outputs(outputs)))


def score_split(project_dir: str | Path, split: str, run_dir: str | Path | None = None,
                outputs: str | Path | None = None, keep_outputs: Path | None = None) -> dict:
    """Produce outputs for one split and evaluate them: one run, one score.

    `run_dir` is where infer.py, src/ and model/ live; it defaults to the project itself,
    and is an experiment snapshot when scoring a past submission. `outputs` skips the first
    step and evaluates a file that already exists. `keep_outputs` saves what the model produced.
    """
    task = load_task(project_dir)
    run_dir = Path(run_dir) if run_dir else task["_project_dir"]
    reference = _reference(task, split)

    with tempfile.TemporaryDirectory() as tmp:
        produced = Path(tmp) / "outputs.csv"
        if outputs is not None:
            produced = Path(outputs).resolve()
            if not produced.exists():
                raise SubmissionError(f"no file at {produced}")
        else:
            _produce(task, split, run_dir, reference, produced)
        if keep_outputs is not None:
            keep_outputs.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced, keep_outputs)
        if evaluator_script(task) is not None:
            scratch = Path(tmp) / "evaluation"
            scratch.mkdir()
            staged = scratch / "outputs.csv"
            staged.symlink_to(produced)
            metrics, n = _evaluator_scores(task, split, reference, staged)
        else:
            metrics, n = _builtin_scores(task, reference, produced)

    return {"split": split, "n": n, "primary": task["metric"]["primary"], "metrics": metrics}

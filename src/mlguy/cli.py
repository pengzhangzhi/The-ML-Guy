"""mlguy: the fixed parts of a project, so the agent can be free with everything else."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

TEMPLATES = Path(__file__).parent / "templates"

from mlguy import budget, report
from mlguy.judge import SubmissionError
from mlguy.project import (
    TaskChanged,
    check_task_unchanged,
    load_task,
    score_split,
    state_dir,
    store_test_labels,
    test_labels_path,
    validation_caveat,
)
from mlguy.records import Experiments, git_commit, hash_task, now

REQUIRED_KEYS = [("name",), ("task", "type"), ("task", "input"), ("task", "output"),
                 ("data", "train"), ("data", "validation"), ("data", "sequence_column"),
                 ("data", "target_column"), ("metric", "primary")]


class CommandError(Exception):
    """Something the user or agent must fix; printed without a traceback."""


def _check_keys(task: dict) -> None:
    missing = []
    for path in REQUIRED_KEYS:
        node = task
        for key in path:
            if not isinstance(node, dict) or key not in node:
                missing.append(".".join(path))
                break
            node = node[key]
    if missing:
        raise CommandError(f"task.toml is missing: {', '.join(missing)}")


def _freeze_task(task: dict) -> str:
    digest = hash_task(task["_project_dir"])
    path = state_dir(task) / "task_hash.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(digest)
    return digest


def _record_score(experiments: Experiments, task: dict, number: int, note: str,
                  counts: bool, result: dict | None, error: str | None) -> dict:
    record = {
        "number": number,
        "time": now(),
        "note": note,
        "status": "ok" if result else "error",
        "counts_against_budget": counts,
    }
    if result:
        record["metrics"] = result["metrics"]
        record["n_validation"] = result["n"]
    else:
        record["error"] = error
    experiments.append(record)
    return record


def _print_scored(record: dict, task: dict, experiments: Experiments) -> None:
    primary = task["metric"]["primary"]
    metrics = "  ".join(f"{k}={v:.4f}" for k, v in record["metrics"].items())
    print(f"experiment {record['number']}: {metrics}  (validation, n={record['n_validation']})")
    best = experiments.best(primary)
    if best and best["number"] != record["number"]:
        print(f"best so far: experiment {best['number']} with {primary}={best['metrics'][primary]:.4f}")
    else:
        print(f"best so far: this one")
    print(budget.describe(task, experiments.all()))


def cmd_new(args: argparse.Namespace) -> int:
    target = Path(args.dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise CommandError(f"{target} already exists and is not empty")
    shutil.copytree(TEMPLATES / "project", target)
    shutil.copy2(TEMPLATES / "proposal.md", target / "proposal.md")

    toml_path = target / "task" / "task.toml"
    toml_path.write_text(toml_path.read_text().replace("TODO-project-name", args.name))
    brief = target / "task" / "brief.md"
    brief.write_text(brief.read_text().replace("TODO name", args.name))
    (target / ".gitignore").write_text("model/\nwork/\n__pycache__/\n.gradio/\n")
    subprocess.run(["git", "init", "-q"], cwd=target, capture_output=True)

    print(f"created {target}")
    print("  task/task.toml   the task: fill in every TODO, then agree it with the user")
    print("  task/brief.md    the one page the user approves")
    print("  predict.py       the fixed interface; src/train.py the baseline")
    print("next: write task/data/{train,validation,test}.csv, then `mlguy init`")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    task = load_task(args.project)
    _check_keys(task)
    project_dir, data = task["_project_dir"], task["data"]
    seq_col, tgt_col = data["sequence_column"], data["target_column"]

    for key in ("train", "validation"):
        path = project_dir / "task" / data[key]
        if not path.exists():
            raise CommandError(f"{key} data not found at {path}")
        columns = pd.read_csv(path, nrows=1).columns
        for needed in (seq_col, tgt_col):
            if needed not in columns:
                raise CommandError(f"{path.name} has no '{needed}' column; found {list(columns)}")

    labeled_test = project_dir / "task" / data["test"] if "test" in data else None
    inputs_path = project_dir / "task" / data.get("test_inputs", "data/test_inputs.csv")
    if labeled_test and labeled_test.exists():
        df = pd.read_csv(labeled_test)
        if tgt_col not in df.columns:
            raise CommandError(f"{labeled_test.name} has no '{tgt_col}' column to hide")
        df[[seq_col]].to_csv(inputs_path, index=False)
        stored = store_test_labels(task, df[[tgt_col]])
        labeled_test.unlink()
        print(f"test labels moved out of the project: {stored}")
    elif not inputs_path.exists():
        raise CommandError(f"no test data: expected {labeled_test or inputs_path}")
    elif not test_labels_path(task).exists():
        raise CommandError(f"test inputs exist but no labels are stored at {test_labels_path(task)}")

    Experiments(project_dir).create()
    digest = _freeze_task(task)
    print(f"project '{task['name']}' ready")
    print(f"  train {len(pd.read_csv(project_dir / 'task' / data['train']))}  "
          f"validation {len(pd.read_csv(project_dir / 'task' / data['validation']))}  "
          f"test {len(pd.read_csv(inputs_path))}")
    print(f"  metric {task['metric']['primary']}  |  {budget.describe(task, [])}")
    print(f"  task frozen (hash {digest[:12]})")
    caveat = validation_caveat(task)
    if caveat:
        print(f"  note: {caveat}")
    print("next: train a model into model/, then run `mlguy validate`")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    task = load_task(args.project)
    check_task_unchanged(task)
    experiments = Experiments(task["_project_dir"])
    if any(r["number"] == 0 for r in experiments.all()) and not args.again:
        raise CommandError("the baseline is already recorded as experiment 0; use --again to redo it")

    try:
        result = score_split(args.project, "validation")
    except SubmissionError as e:
        print(f"the pipeline does not work yet:\n{e}", file=sys.stderr)
        return 1

    number = 0 if args.again else experiments.next_number()
    experiments.snapshot(task["_project_dir"], number)
    record = _record_score(experiments, task, number, args.note, False, result, None)
    print("prediction and scoring work end to end.")
    _print_scored(record, task, experiments)
    print("the loop is open: change the modelling however you like, then `mlguy submit --note \"...\"`")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    task = load_task(args.project)
    check_task_unchanged(task)
    experiments = Experiments(task["_project_dir"])
    records = experiments.all()
    if not any(r["number"] == 0 for r in records):
        raise CommandError("run `mlguy validate` first, so the baseline is on record")

    status = budget.state(task, records)
    if status["spent"]:
        print(f"refused: {status['reason']}. Run `mlguy finish`.", file=sys.stderr)
        return 2

    number = experiments.next_number()
    try:
        result, error = score_split(args.project, "validation"), None
    except SubmissionError as e:
        result, error = None, str(e)

    target, size_mb = experiments.snapshot(task["_project_dir"], number)
    if size_mb > 200:
        print(f"note: the snapshot of this submission is {size_mb:.0f} MB", file=sys.stderr)
    record = _record_score(experiments, task, number, args.note, True, result, error)
    commit = git_commit(task["_project_dir"], f"experiment {number}: {args.note}")
    if commit:
        record["commit"] = commit

    if error:
        print(f"experiment {number} did not score:\n{error}", file=sys.stderr)
        print("recorded as an error, not as a zero. " + budget.describe(task, experiments.all()), file=sys.stderr)
        return 1
    _print_scored(record, task, experiments)
    return 0


def cmd_finish(args: argparse.Namespace) -> int:
    task = load_task(args.project)
    check_task_unchanged(task)
    project_dir = task["_project_dir"]
    experiments = Experiments(project_dir)
    records = experiments.all()
    primary = task["metric"]["primary"]
    best = experiments.best(primary)
    if best is None:
        raise CommandError("nothing has scored yet, so there is no model to report")

    final_path = state_dir(task) / "test_score.json"
    if final_path.exists() and not args.again:
        previous = json.loads(final_path.read_text())
        raise CommandError(
            f"the test set was already scored (experiment {previous['number']}, "
            f"{primary}={previous['metrics'][primary]:.4f}). Scoring it again would let the search fit to it; "
            "use --again only if you know why that is safe."
        )

    snapshot = experiments.dir / f"{best['number']:04d}" / "submission"
    if not snapshot.exists():
        raise CommandError(f"no saved submission for experiment {best['number']} at {snapshot}")
    try:
        test_result = score_split(project_dir, "test", run_dir=snapshot)
    except SubmissionError as e:
        raise CommandError(f"the best submission failed on the test set:\n{e}")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps({"number": best["number"], "time": now(), **test_result}, indent=2))
    path = report.write(project_dir, task, records, best, test_result)
    git_commit(project_dir, f"finish: test {primary}={test_result['metrics'][primary]:.4f}")
    print(f"test {primary} = {test_result['metrics'][primary]:.4f}  (n={test_result['n']}, scored once)")
    print(f"report written to {path}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    task = load_task(args.project)
    experiments = Experiments(task["_project_dir"])
    records = experiments.all()
    primary = task["metric"]["primary"]
    best = experiments.best(primary)

    if args.json:
        final = state_dir(task) / "test_score.json"
        print(json.dumps({
            "project": task["name"],
            "metric": primary,
            "experiments": len(records),
            "best": best,
            "budget": budget.state(task, records),
            "test_scored": json.loads(final.read_text()) if final.exists() else None,
        }, indent=2))
        return 0

    print(f"project: {task['name']}  |  predicting {task['task']['output']}  |  metric {primary}")
    if not records:
        print("nothing recorded yet; train a model into model/ and run `mlguy validate`")
        return 0
    for record in records[-args.last:]:
        if record["status"] == "ok":
            metrics = "  ".join(f"{k}={v:.4f}" for k, v in record["metrics"].items())
        else:
            metrics = f"error: {record['error'].splitlines()[0][:60]}"
        mark = "*" if best and record["number"] == best["number"] else " "
        print(f" {mark}{record['number']:>4}  {metrics:<34}  {record.get('note', '')[:50]}")
    if best:
        print(f"best: experiment {best['number']} with {primary}={best['metrics'][primary]:.4f}  (* above)")
    print(budget.describe(task, records))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    try:
        result = score_split(args.project, args.split)
    except SubmissionError as e:
        print(json.dumps({"status": "error", "split": args.split, "error": str(e)}, indent=2))
        return 1
    print(json.dumps({"status": "ok", **result}, indent=2))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    page = Path(__file__).parent / "dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(page),
           "--server.headless", "true", "--server.port", str(args.port), "--", str(Path(args.project).resolve())]
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        raise CommandError("streamlit is not installed; `pip install 'mlguy[dashboard]'`")


def cmd_serve(args: argparse.Namespace) -> int:
    from mlguy.app import launch

    launch(args.project, port=args.port, share=args.share)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlguy", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, help_, func):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--project", default=".", help="project folder (default: current directory)")
        p.set_defaults(func=func)
        return p

    p = sub.add_parser("new", help="create an empty project from the template")
    p.add_argument("dir", help="folder to create")
    p.add_argument("--name", required=True, help="short kebab-case project name")
    p.set_defaults(func=cmd_new)

    p = add("init", "check the task, hide the test labels, freeze task/", cmd_init)

    p = add("validate", "check that prediction and scoring work; records the baseline as experiment 0", cmd_validate)
    p.add_argument("--note", default="baseline", help="one line on what this model is")
    p.add_argument("--again", action="store_true", help="redo the baseline")

    p = add("submit", "score the current model on validation, record it, enforce the budget", cmd_submit)
    p.add_argument("--note", required=True, help="one line on what you changed and why")

    p = add("finish", "score the best submission on the test set once and write the report", cmd_finish)
    p.add_argument("--again", action="store_true", help="score the test set again (it fits the search to it)")

    p = add("status", "what has been tried, what is best, what budget is left", cmd_status)
    p.add_argument("--json", action="store_true")
    p.add_argument("--last", type=int, default=20, help="how many experiments to list")

    p = add("score", "run predict.py on a split and score it, without recording anything", cmd_score)
    p.add_argument("--split", choices=["validation", "test"], default="validation")

    p = add("dashboard", "watch the experiments in a browser", cmd_dashboard)
    p.add_argument("--port", type=int, default=8501)

    p = add("serve", "launch the prediction page", cmd_serve)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--share", action="store_true", help="create a public Gradio link")

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (CommandError, TaskChanged, FileNotFoundError) as e:
        print(f"{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

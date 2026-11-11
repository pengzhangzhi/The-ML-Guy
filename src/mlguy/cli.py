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

from mlguy import budget, checks, judge, report
from mlguy.judge import SubmissionError
from mlguy.project import (
    TaskChanged,
    check_task_unchanged,
    compare_with,
    direction,
    entry_point,
    evaluator_script,
    is_generation,
    sampling,
    load_task,
    score_split,
    state_dir,
    store_test_labels,
    test_labels_path,
    validation_caveat,
)
from mlguy.records import Experiments, git_commit, hash_task, now

REQUIRED_KEYS = [("name",), ("task", "type"), ("task", "output"), ("data", "train"), ("metric", "primary")]
# a built-in metric scores one prediction per row against a label column
BUILTIN_KEYS = [("data", "validation"), ("data", "input_columns"), ("data", "target_column")]
# the task's own evaluator names its metric, so the task must say which way is better
EVALUATOR_KEYS = [("metric", "direction")]


class CommandError(Exception):
    """Something the user or agent must fix; printed without a traceback."""


def _check_keys(task: dict) -> None:
    missing = []
    custom = "script" in task.get("evaluator", {})
    if task.get("task", {}).get("type") == "generation" and not custom:
        raise CommandError("a generation task needs its own evaluator: set [evaluator] script in task.toml")
    for path in REQUIRED_KEYS + (EVALUATOR_KEYS if custom else BUILTIN_KEYS):
        node = task
        for key in path:
            if not isinstance(node, dict) or key not in node:
                missing.append(".".join(path))
                break
            node = node[key]
    if missing:
        raise CommandError(f"task.toml is missing: {', '.join(missing)}")
    primary = task["metric"]["primary"]
    if custom:
        if task["metric"]["direction"] not in ("maximize", "minimize"):
            raise CommandError("metric.direction must be 'maximize' or 'minimize'")
        if not evaluator_script(task).exists():
            raise CommandError(f"the evaluator {evaluator_script(task)} does not exist")
        if task["data"].get("input_columns") and "validation" not in task["data"]:
            raise CommandError("task.toml is missing: data.validation (the rows to produce outputs for)")
    elif primary not in judge.METRICS:
        raise CommandError(f"unknown metric '{primary}'; built in: {', '.join(sorted(judge.METRICS))}. "
                           "For anything else, give the task its own evaluator.")
    elif judge.METRICS[primary][2] != task["task"]["type"]:
        raise CommandError(f"'{primary}' is a {judge.METRICS[primary][2]} metric, but the task is {task['task']['type']}")


def _freeze_task(task: dict) -> str:
    digest = hash_task(task["_project_dir"])
    path = state_dir(task) / "task_hash.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(digest)
    return digest


def _record_score(experiments: Experiments, task: dict, number: int, note: str,
                  counts: bool, result: dict | None, error: str | None,
                  previous_best: dict | None = None) -> dict:
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
        record["interval"] = result.get("interval")
        if previous_best and previous_best["number"] != number:
            chance = compare_with(task["_project_dir"], experiments.predictions(number),
                                  experiments.predictions(previous_best["number"]))
            if chance is not None:
                record["vs_best"] = {"number": previous_best["number"], "chance_better": round(chance, 3)}
    else:
        record["error"] = error
    if any(r["number"] == number for r in experiments.all()):
        experiments.replace(record)
    else:
        experiments.append(record)
    return record


def _print_scored(record: dict, task: dict, experiments: Experiments) -> None:
    primary = task["metric"]["primary"]
    metrics = "  ".join(f"{k}={v:.4f}" for k, v in record["metrics"].items())
    print(f"experiment {record['number']}: {metrics}  (validation, n={record['n_validation']})")
    if record.get("interval"):
        low, high = record["interval"]
        print(f"  {primary} would fall between {low:.4f} and {high:.4f} on another draw of the validation rows")
    if record.get("vs_best"):
        versus = record["vs_best"]
        print(f"  against experiment {versus['number']}, the best before this: better on "
              f"{versus['chance_better']:.0%} of resamples: {judge.verdict(versus['chance_better'])}")
    best = experiments.best(primary, direction(task))
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
    in_cols, tgt_col = list(data.get("input_columns", [])), data.get("target_column")

    frozen = state_dir(task) / "task_hash.txt"
    if frozen.exists() and Experiments(project_dir).all() and frozen.read_text().strip() != hash_task(project_dir):
        raise CommandError(
            "task/ has changed since it was frozen, and this project already has recorded experiments. "
            "Their scores were measured against the old task, so it cannot be re-frozen. Restore task/, "
            "or start a new project for the new task."
        )

    builtin, generation = evaluator_script(task) is None, is_generation(task)
    needed = [*in_cols, tgt_col] if builtin else in_cols
    for key in ("train", "validation"):
        if key not in data:
            continue
        path = project_dir / "task" / data[key]
        if not path.exists():
            raise CommandError(f"{key} data not found at {path}")
        columns = pd.read_csv(path, nrows=1).columns
        for column in needed:
            if column not in columns:
                raise CommandError(f"{path.name} has no '{column}' column; found {list(columns)}")

    # Hold back the test rows. Prediction keeps the inputs in the project and hides the rest;
    # generation hides the rows entirely, so the final conditions are unseen until `finish`.
    labeled_test = project_dir / "task" / data["test"] if "test" in data else None
    inputs_path = project_dir / "task" / data.get("test_inputs", "data/test_inputs.csv")
    test_frame = None
    if labeled_test and labeled_test.exists():
        test_frame = pd.read_csv(labeled_test)
        for column in needed:
            if column not in test_frame.columns:
                raise CommandError(f"{labeled_test.name} has no '{column}' column")
        if not generation:
            test_frame[in_cols].to_csv(inputs_path, index=False)
        stored = store_test_labels(task, test_frame)
        labeled_test.unlink()
        print(f"test {'rows' if generation else 'labels'} moved out of the project: {stored}")
    elif test_labels_path(task).exists():
        test_frame = pd.read_csv(test_labels_path(task))
        if not generation and not inputs_path.exists():
            raise CommandError(f"test labels are stored but the test inputs are missing at {inputs_path}")
        if not generation and not set(in_cols) <= set(test_frame.columns):   # stored by an older version
            test_frame = pd.concat([pd.read_csv(inputs_path), test_frame], axis=1)
    elif not generation:
        raise CommandError(f"no test data: expected {labeled_test or 'data.test in task.toml'}")

    train_frame = pd.read_csv(project_dir / "task" / data["train"])
    validation_frame = pd.read_csv(project_dir / "task" / data["validation"]) if "validation" in data else None
    found = checks.split_checks(task, train_frame, validation_frame, test_frame, compare_targets=builtin)
    (state_dir(task)).mkdir(parents=True, exist_ok=True)
    (state_dir(task) / "split_checks.json").write_text(json.dumps(found, indent=2))

    Experiments(project_dir).create()
    digest = _freeze_task(task)
    print(f"project '{task['name']}' ready")
    sizes = [("train", train_frame), ("validation", validation_frame), ("test", test_frame)]
    print("  " + "  ".join(f"{name} {len(frame)}" for name, frame in sizes if frame is not None))
    if generation:
        (n, seed), (final_n, final_seed) = sampling(task, "validation"), sampling(task, "test")
        per = " per condition" if in_cols else ""
        print(f"  sampling: {n}{per} on every submission (seed {seed}); {final_n}{per} for the final "
              f"check (seed {final_seed}" + (", held-out conditions)" if test_frame is not None else ")"))
    scored_by = "built in" if builtin else f"task/{task['evaluator']['script']}"
    print(f"  metric {task['metric']['primary']} ({direction(task)}, {scored_by})  |  {budget.describe(task, [])}")
    print(f"  task frozen (hash {digest[:12]})")
    caveat = validation_caveat(task)
    if caveat:
        print(f"  note: {caveat}")
    for item in found:
        print(f"  check: {item}")
    if found:
        print("  fix each check above, or explain it in task/brief.md; they are repeated in the final report")
    print(f"next: train a model into model/, make {entry_point(task)} work, then run `mlguy validate`")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    task = load_task(args.project)
    check_task_unchanged(task)
    experiments = Experiments(task["_project_dir"])
    if any(r["number"] == 0 for r in experiments.all()) and not args.again:
        raise CommandError("the baseline is already recorded as experiment 0; use --again to redo it")
    if any(r.get("counts_against_budget") for r in experiments.all()):
        raise CommandError("the baseline cannot be redone once submissions exist: it would be a "
                           "submission the budget does not count. Use `mlguy submit`.")

    try:
        result = score_split(args.project, "validation", keep_predictions=experiments.predictions(0))
    except SubmissionError as e:
        print(f"the pipeline does not work yet:\n{e}", file=sys.stderr)
        return 1

    number = 0
    experiments.snapshot(task["_project_dir"], number)
    record = _record_score(experiments, task, number, args.note, False, result, None)
    print(f"{entry_point(task)} and scoring work end to end.")
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
    previous_best = experiments.best(task["metric"]["primary"], direction(task))
    try:
        result, error = score_split(args.project, "validation",
                                    keep_predictions=experiments.predictions(number)), None
    except SubmissionError as e:
        result, error = None, str(e)

    target, size_mb = experiments.snapshot(task["_project_dir"], number)
    if size_mb > 200:
        print(f"note: the snapshot of this submission is {size_mb:.0f} MB", file=sys.stderr)
    record = _record_score(experiments, task, number, args.note, True, result, error, previous_best)
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
    best = experiments.best(primary, direction(task))
    if best is None:
        raise CommandError("nothing has scored yet, so there is no model to report")

    final_path = state_dir(task) / "test_score.json"
    if final_path.exists() and not args.again:
        previous = json.loads(final_path.read_text())
        raise CommandError(
            f"the final score was already taken (experiment {previous['number']}, "
            f"{primary}={previous['metrics'][primary]:.4f}). Taking it again would let the search fit to it; "
            "use --again only if you know why that is safe."
        )

    snapshot = experiments.dir / f"{best['number']:04d}" / "submission"
    if not snapshot.exists():
        raise CommandError(f"no saved submission for experiment {best['number']} at {snapshot}")
    try:
        test_result = score_split(project_dir, "test", run_dir=snapshot)
    except SubmissionError as e:
        raise CommandError(f"the best submission failed the final check:\n{e}")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps({"number": best["number"], "time": now(), **test_result}, indent=2))
    path = report.write(project_dir, task, records, best, test_result)
    git_commit(project_dir, f"finish: test {primary}={test_result['metrics'][primary]:.4f}")
    label = "final" if is_generation(task) else "test"
    print(f"{label} {primary} = {test_result['metrics'][primary]:.4f}  (n={test_result['n']}, scored once)")
    if test_result.get("interval"):
        print(f"  it would fall between {test_result['interval'][0]:.4f} and {test_result['interval'][1]:.4f} "
              "on another draw")
    print(f"report written to {path}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    task = load_task(args.project)
    experiments = Experiments(task["_project_dir"])
    records = experiments.all()
    primary = task["metric"]["primary"]
    best = experiments.best(primary, direction(task))

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

    verb = "generating" if is_generation(task) else "predicting"
    print(f"project: {task['name']}  |  {verb} {task['task']['output']}  |  metric {primary}")
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
        result = score_split(args.project, "validation", outputs=args.outputs)
    except SubmissionError as e:
        print(json.dumps({"status": "error", "split": "validation", "error": str(e)}, indent=2))
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
    """Run the project's own page. The framework ships none: the agent builds it for the problem."""
    project_dir = Path(args.project).resolve()
    if not (project_dir / "app.py").exists():
        raise CommandError(f"no app.py in {project_dir}: this project's page has not been built yet "
                           "(the deploy skill covers it)")
    cmd = [sys.executable, "app.py"]
    if args.port:
        cmd += ["--port", str(args.port)]
    if args.share:
        cmd += ["--share"]
    return subprocess.call(cmd, cwd=project_dir)


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
    p.add_argument("--again", action="store_true", help="redo the baseline (only before the first submission)")

    p = add("submit", "score the current model on validation, record it, enforce the budget", cmd_submit)
    p.add_argument("--note", required=True, help="one line on what you changed and why")

    p = add("finish", "score the best submission on the test set once and write the report", cmd_finish)
    p.add_argument("--again", action="store_true", help="score the test set again (it fits the search to it)")

    p = add("status", "what has been tried, what is best, what budget is left", cmd_status)
    p.add_argument("--json", action="store_true")
    p.add_argument("--last", type=int, default=20, help="how many experiments to list")

    p = add("score", "score the current model on validation without recording it (the final check is "
                     "only ever run by `finish`)", cmd_score)
    p.add_argument("--outputs", default=None,
                   help="evaluate this file of outputs instead of running the model (to try out an evaluator)")

    p = add("dashboard", "watch the experiments in a browser", cmd_dashboard)
    p.add_argument("--port", type=int, default=8501)

    p = add("serve", "run the project's own page (app.py)", cmd_serve)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--share", action="store_true", help="passed to app.py as --share")

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (CommandError, TaskChanged, FileNotFoundError) as e:
        print(f"{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

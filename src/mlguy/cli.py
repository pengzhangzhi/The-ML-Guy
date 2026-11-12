"""mlguy: the fixed parts of a project, so the agent can be free with everything else."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

TEMPLATES = Path(__file__).parent / "templates"

from mlguy import budget, checks, report
from mlguy.judge import SubmissionError
from mlguy.project import (
    TaskChanged,
    check_task_unchanged,
    task_changes,
    direction,
    ENTRY_POINT,
    data_path,
    hidden_inputs_path,
    id_column,
    target_column,
    evaluator_script,
    is_generation,
    sampling,
    STATE_POINTER,
    mlguy_home,
    load_task,
    score_split,
    state_dir,
    test_labels_path,
    validation_caveat,
)
from mlguy.records import Experiments, git_commit, hash_task, now, task_manifest

REQUIRED_KEYS = [("name",), ("task", "type"), ("task", "output"), ("data", "train"),
                 ("metric", "primary"), ("metric", "direction")]
# prediction: each split is inputs (any format) plus a table of ids and labels
PREDICTION_KEYS = [("data", "validation_inputs"), ("data", "validation_labels"),
                   ("data", "test_inputs"), ("data", "test_labels")]
APPROVAL = "Approved by the user:"


class CommandError(Exception):
    """Something the user or agent must fix; printed without a traceback."""


def _check_keys(task: dict) -> None:
    missing = []
    generation = task.get("task", {}).get("type") == "generation"
    for path in REQUIRED_KEYS + ([] if generation else PREDICTION_KEYS):
        node = task
        for key in path:
            if not isinstance(node, dict) or key not in node:
                missing.append(".".join(path))
                break
            node = node[key]
    if missing:
        raise CommandError(f"task.toml is missing: {', '.join(missing)}")
    if task["metric"]["direction"] not in ("maximize", "minimize"):
        raise CommandError("metric.direction must be 'maximize' or 'minimize'")
    if not evaluator_script(task).exists():
        raise CommandError(f"no evaluator at {evaluator_script(task)}: every task carries its own "
                           "(the build-project skill covers writing it)")
    for inputs, labels in (("validation_inputs", "validation_labels"), ("test_inputs", "test_labels")):
        if inputs in task["data"] and labels not in task["data"]:
            raise CommandError(f"task.toml has data.{inputs} but no data.{labels} (the ids to produce outputs for)")


def _freeze_task(task: dict) -> str:
    state_dir(task).mkdir(parents=True, exist_ok=True)
    (state_dir(task) / "project_path.txt").write_text(str(task["_project_dir"]))
    # remember where the state is, so MLGUY_HOME need not be set on every later command
    (task["_project_dir"] / STATE_POINTER).write_text(str(mlguy_home(task["_project_dir"])) + "\n")
    manifest = task_manifest(task["_project_dir"])
    (state_dir(task) / "task_files.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return hash_task(task["_project_dir"])


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
    if any(r["number"] == number for r in experiments.all()):
        experiments.replace(record)
    else:
        experiments.append(record)
    return record


def _print_scored(record: dict, task: dict, experiments: Experiments,
                  previous_best: dict | None = None) -> None:
    primary = task["metric"]["primary"]
    metrics = "  ".join(f"{k}={v:.4f}" for k, v in record["metrics"].items())
    print(f"experiment {record['number']}: {metrics}  (validation, n={record['n_validation']})")
    if previous_best and previous_best["number"] != record["number"]:
        change = record["metrics"][primary] - previous_best["metrics"][primary]
        print(f"  {change:+.4f} {primary} against experiment {previous_best['number']}, the best before this")
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

    toml_path = target / "task" / "task.toml"
    toml_path.write_text(toml_path.read_text().replace("TODO-project-name", args.name))
    brief = target / "task" / "brief.md"
    brief.write_text(brief.read_text().replace("TODO name", args.name))
    # data is frozen by its hash, not by git, so large inputs do not end up in every clone
    (target / ".gitignore").write_text("model/\n!final/model/\nwork/\n__pycache__/\ntask/data/\n")
    subprocess.run(["git", "init", "-q"], cwd=target, capture_output=True)

    print(f"created {target}")
    print("  task/task.toml   the task: fill in every TODO, then agree it with the user")
    print("  task/brief.md    the one page the user approves")
    print("  this folder is a git repository: `validate`, `submit` and `finish` commit to it (not task/data)")
    print("  task/evaluate.py the scoring, as a stub: write it for the problem; it is frozen with the task")
    print("  infer.py         the one fixed interface, as a stub; src/ is yours")
    print("next: prepare task/data, write the evaluator and the simplest model that runs, get the brief "
          "approved, then `mlguy init`")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    task = load_task(args.project)
    _check_keys(task)
    project_dir, data = task["_project_dir"], task["data"]
    id_col, tgt_col = id_column(task), target_column(task)
    generation = is_generation(task)
    brief = project_dir / "task" / "brief.md"
    approval = [line for line in (brief.read_text().splitlines() if brief.exists() else [])
                if line.strip().startswith(APPROVAL)]
    if not approval or not approval[0].strip()[len(APPROVAL):].strip(" -:*_") or "TODO" in approval[0]:
        raise CommandError(f"task/brief.md does not record the user's approval. Show them the brief, and "
                           f"once they approve, add a line '{APPROVAL} <their words, and the date>'.")

    owner = state_dir(task) / "project_path.txt"
    if owner.exists() and owner.read_text().strip() != str(project_dir):
        raise CommandError(
            f"another project called '{task['name']}' already keeps its state in {state_dir(task)} "
            f"(it lives at {owner.read_text().strip()}). Give this project a different name in task.toml."
        )

    frozen = state_dir(task) / "task_files.json"
    if frozen.exists() and Experiments(project_dir).all() and task_changes(task):
        raise CommandError(
            f"task/ has changed since it was frozen ({', '.join(task_changes(task)[:3])}), and this project "
            "already has recorded experiments. "
            "Their scores were measured against the old task, so it cannot be re-frozen. Restore task/, "
            "or start a new project for the new task."
        )

    def labels_table(path: Path, name: str) -> pd.DataFrame:
        frame = pd.read_csv(path, dtype={id_col: str})
        for column in [id_col]:
            if column not in frame.columns:
                raise CommandError(f"{path.name} has no '{column}' column; found {list(frame.columns)}")
        if frame[id_col].duplicated().any():
            raise CommandError(f"{path.name}: {int(frame[id_col].duplicated().sum())} ids are repeated; "
                               f"every {name} example needs its own id")
        return frame

    for key in ("train", "validation_inputs", "validation_labels"):
        if key in data and not data_path(task, key).exists():
            raise CommandError(f"{key} not found at {data_path(task, key)}")
    validation_frame = labels_table(data_path(task, "validation_labels"), "validation") \
        if "validation_labels" in data else None

    # Hold back the test set. Prediction keeps the inputs in the project and hides the labels;
    # generation hides the inputs too, so the final conditions are unseen until `finish`.
    test_labels, test_inputs, test_frame = data_path(task, "test_labels"), data_path(task, "test_inputs"), None
    if test_labels is not None and test_labels.exists():
        if not test_inputs.exists():
            raise CommandError(f"test_inputs not found at {test_inputs}")
        test_frame = labels_table(test_labels, "test")
        test_labels_path(task).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(test_labels, test_labels_path(task))
        if generation:
            shutil.move(test_inputs, hidden_inputs_path(task))
        print(f"test {'set' if generation else 'labels'} moved out of the project: {test_labels_path(task).parent}")
        print("  anything else that still holds those answers — the file your preparation script "
              "read — belongs outside the project too; training code can open any path")
    elif test_labels_path(task).exists():
        test_frame = pd.read_csv(test_labels_path(task), dtype={id_col: str})
        if not generation and not test_inputs.exists():
            raise CommandError(f"test labels are stored but the test inputs are missing at {test_inputs}")
    elif not generation:
        raise CommandError(f"no test labels: expected {test_labels}")

    state_dir(task).mkdir(parents=True, exist_ok=True)
    stored = {"train": data_path(task, "train"), "validation": data_path(task, "validation_inputs"),
              "test": None if generation else test_inputs}
    both_labelled = all(f is not None and tgt_col in f.columns for f in (validation_frame, test_frame))
    found = checks.split_checks(task, validation_frame, test_frame, stored, id_col, tgt_col,
                                compare_targets=both_labelled)
    (state_dir(task) / "split_checks.json").write_text(json.dumps(found, indent=2))

    Experiments(project_dir).create()
    digest = _freeze_task(task)
    print(f"project '{task['name']}' ready")
    sizes = [("validation", validation_frame), ("test", test_frame)]
    print(f"  train {data['train']}  " + "  ".join(f"{name} {len(frame)}" for name, frame in sizes if frame is not None))
    if generation:
        (n, seed), (final_n, final_seed) = sampling(task, "validation"), sampling(task, "test")
        per = " per condition" if validation_frame is not None else ""
        print(f"  sampling: {n}{per} on every submission (seed {seed}); {final_n}{per} for the final "
              f"check (seed {final_seed}" + (", held-out conditions)" if test_frame is not None else ")"))
    scored_by = f"task/{evaluator_script(task).name}"
    print(f"  metric {task['metric']['primary']} ({direction(task)}, {scored_by})  |  {budget.describe(task, [])}")
    print(f"  task frozen (hash {digest[:12]})")
    caveat = validation_caveat(task)
    if caveat:
        print(f"  note: {caveat}")
    if not (project_dir / "task" / "audit.md").exists():
        print("  note: no task/audit.md, so no fresh agent has tried to break the split and the evaluator "
              "(the audit skill)")
    for item in found:
        print(f"  check: {item}")
    if found:
        print("  fix each check above, or explain it in task/brief.md; they are repeated in the final report")
    skipped = []
    opaque = [name for name, path in stored.items()
              if path is not None and not (path.suffix == ".csv" and path.is_file())]
    if opaque and not generation:
        skipped.append(f"whether the same example appears in two splits, because the {' and '.join(opaque)} "
                       "inputs are not CSV tables the framework can read; check this yourself")
    if not both_labelled and not generation:
        skipped.append(f"the label distributions of validation and test were not compared, because the "
                       f"labels files have no '{tgt_col}' column (set data.target_column if it has another name)")
    elif not task["data"].get("validation_matches_test", True):
        skipped.append("the label distributions of validation and test were not compared, because the "
                       "difference is already on record (validation_matches_test = false)")
    for item in skipped:
        print(f"  not checked: {item}")
    if not found:
        print("  split checks: nothing raised")
    print(f"next: train a model into model/, make {ENTRY_POINT} work, then run `mlguy validate`")
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
        result = score_split(args.project, "validation", keep_outputs=experiments.outputs(0))
    except SubmissionError as e:
        print(f"the pipeline does not work yet:\n{e}", file=sys.stderr)
        return 1

    number = 0
    experiments.snapshot(task["_project_dir"], number)
    record = _record_score(experiments, task, number, args.note, False, result, None)
    git_commit(task["_project_dir"], f"experiment 0 (starting model): {args.note}")
    print(f"{ENTRY_POINT} and scoring work end to end.")
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
                                    keep_outputs=experiments.outputs(number)), None
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
    _print_scored(record, task, experiments, previous_best)
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
    chosen, reason = best, None
    if args.experiment is not None and args.experiment != best["number"]:
        chosen = next((r for r in experiments.submissions() if r["number"] == args.experiment), None)
        if chosen is None:
            raise CommandError(f"experiment {args.experiment} has no score, so it cannot be the result")
        if not args.reason:
            raise CommandError(f"experiment {best['number']} scored best on validation; say why "
                               f"experiment {args.experiment} should be the result instead, with --reason")
        reason = args.reason

    final_path = state_dir(task) / "test_score.json"
    if final_path.exists() and not args.again:
        previous = json.loads(final_path.read_text())
        raise CommandError(
            f"the final score was already taken (experiment {previous['number']}, "
            f"{primary}={previous['metrics'][primary]:.4f}). Taking it again would let the search fit to it; "
            "use --again only if you know why that is safe."
        )

    snapshot = experiments.dir / f"{chosen['number']:04d}" / "submission"
    if not snapshot.exists():
        raise CommandError(f"no saved submission for experiment {chosen['number']} at {snapshot}")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "test_outputs.csv"
            test_result = score_split(project_dir, "test", run_dir=snapshot, keep_outputs=outputs)
            final_dir = project_dir / "final"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(snapshot, final_dir)
            shutil.copy2(outputs, final_dir / "test_outputs.csv")
    except SubmissionError as e:
        raise CommandError(f"the best submission failed the final check:\n{e}")

    # The working folder holds whatever was tried last; final/ holds what was scored, and the
    # page and the user run the model from there. The test set is now spent, so its labels go
    # beside the outputs: the report can say what the score means case by case.
    if test_labels_path(task).exists():
        shutil.copy2(test_labels_path(task), final_dir / "test_labels.csv")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps({"number": chosen["number"], "best_on_validation": best["number"],
                                      "reason": reason, "time": now(), **test_result}, indent=2))
    path = report.write(project_dir, task, records, chosen, test_result, best_on_validation=best, reason=reason)
    git_commit(project_dir, f"finish: test {primary}={test_result['metrics'][primary]:.4f}")
    label = "final" if is_generation(task) else "test"
    print(f"{label} {primary} = {test_result['metrics'][primary]:.4f}  (n={test_result['n']}, scored once)")
    print(f"report written to {path}")
    print("final/ holds the scored model, with its test outputs and the test labels beside them")
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

    p = add("validate", "check that infer.py and scoring work; records the baseline as experiment 0", cmd_validate)
    p.add_argument("--note", default="baseline", help="one line on what this model is")
    p.add_argument("--again", action="store_true", help="redo the baseline (only before the first submission)")

    p = add("submit", "score the current model on validation, record it, enforce the budget", cmd_submit)
    p.add_argument("--note", required=True, help="one line on what you changed and why")

    p = add("finish", "score the result on the test set once and write the report", cmd_finish)
    p.add_argument("--again", action="store_true", help="score the test set again (it fits the search to it)")
    p.add_argument("--experiment", type=int, default=None,
                   help="the submission to take as the result, when it is not the best on validation "
                        "(say, a simpler one that cannot be told apart from the best)")
    p.add_argument("--reason", default=None, help="why, when --experiment is not the best on validation")

    p = add("status", "what has been tried, what is best, what budget is left", cmd_status)
    p.add_argument("--json", action="store_true")
    p.add_argument("--last", type=int, default=20, help="how many experiments to list")

    p = add("score", "run the task's scoring on a file of outputs, to check that it tells good from bad "
                     "before the task is frozen. Records nothing.", cmd_score)
    p.add_argument("--outputs", required=True, help="a file of outputs for the validation examples")

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (CommandError, TaskChanged, FileNotFoundError) as e:
        print(f"{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

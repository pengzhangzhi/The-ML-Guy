"""REPORT.md: the final scores, in the language of the brief."""

from __future__ import annotations

import json
from pathlib import Path

from mlguy import budget, judge
from mlguy.project import compare_with, direction, is_generation, sampling, state_dir, validation_caveat
from mlguy.records import Experiments


MAX_COLUMNS = 4  # the experiment table shows the primary metric and a few others; log.jsonl has them all


def _row(record: dict, names: list[str]) -> str:
    note = record.get("note", "").replace("|", "/")
    if record["status"] != "ok":
        return f"| {record['number']} | {note} | error |" + " |" * (len(names) - 1)
    values = " | ".join(f"{record['metrics'][n]:.4f}" if n in record["metrics"] else "" for n in names)
    return f"| {record['number']} | {note} | {values} |"


def _budget_used(task: dict, records: list[dict]) -> str:
    status = budget.state(task, records)
    parts = []
    if "submissions" in status["limits"]:
        s = status["limits"]["submissions"]
        parts.append(f"{s['used']} of {s['limit']} submissions used")
    if "hours" in status["limits"]:
        h = status["limits"]["hours"]
        parts.append(f"{h['used']} of {h['limit']} hours used")
    errors = sum(1 for r in records if r["status"] == "error")
    if errors:
        parts.append(f"{errors} submission(s) failed to run")
    return (", ".join(parts).capitalize() + ".") if parts else "No budget was set."


def _chance_over_baseline(project_dir: Path, best: dict) -> float | None:
    experiments = Experiments(project_dir)
    return compare_with(project_dir, experiments.predictions(best["number"]), experiments.predictions(0))


def _split_checks(task: dict) -> list[str]:
    path = state_dir(task) / "split_checks.json"
    return json.loads(path.read_text()) if path.exists() else []


def write(project_dir: Path, task: dict, records: list[dict], best: dict, test_result: dict,
          best_on_validation: dict | None = None, reason: str | None = None) -> Path:
    """`best` is the submission taken as the result; normally also the best on validation."""
    primary = task["metric"]["primary"]
    declared = [primary] + [m for m in task["metric"].get("secondary", []) if m != primary]
    names = [n for n in declared if n in best["metrics"]] or list(best["metrics"].keys())
    names = names[:MAX_COLUMNS]
    better = "higher" if direction(task) == "maximize" else "lower"
    counted = sum(1 for r in records if r.get("counts_against_budget"))
    limit = task.get("budget", {}).get("max_submissions")
    used = f"{counted} of {limit} submissions" if limit else f"{counted} submissions"
    baseline = next((r for r in records if r["number"] == 0 and r["status"] == "ok"), None)
    output = task["task"]["output"].replace("_", " ")
    given = task["task"].get("input", "").replace("_", " ")
    final_n, final_seed = sampling(task, "test")
    if is_generation(task):
        headline = f"Generates **{output}**" + (f" for {given}." if given else ".")
        measured = [f"Measured on {test_result['n']} samples drawn fresh for the final check (seed {final_seed}),",
                    f"scored once, at the end of the search ({used} used)."]
        columns = "| Metric | Final check | During the search (best submission) |"
    else:
        headline = f"Predicts **{output}** from {given}."
        measured = [f"Measured on {test_result['n']} held-out examples the model never saw during the search,",
                    f"scored once, at the end of the search ({used} used)."]
        columns = "| Metric | Test | Validation (best submission) |"

    lines = [
        f"# {task['name']}: result",
        "",
        headline,
        "",
        "## Score",
        "",
        *measured,
        "",
        columns,
        "| --- | --- | --- |",
    ]
    for name in names:
        test_value = test_result["metrics"].get(name)
        lines.append(
            f"| {name} | n/a | {best['metrics'][name]:.4f} |" if test_value is None else
            f"| {name} | {test_value:.4f} | {best['metrics'][name]:.4f} |"
        )
    lines += ["", f"The score that counts is **{primary} = {test_result['metrics'][primary]:.4f}** "
                  f"({better} is better).", ""]
    if test_result.get("interval"):
        low, high = test_result["interval"]
        lines += [f"On a different draw of {test_result['n']} it would be expected to fall "
                  f"between {low:.4f} and {high:.4f} (95% interval).", ""]

    if best_on_validation and best_on_validation["number"] != best["number"]:
        top = best_on_validation["metrics"][primary]
        lines += [f"The result is experiment {best['number']} ({primary} {best['metrics'][primary]:.4f} on "
                  f"validation), chosen over experiment {best_on_validation['number']}, which scored best "
                  f"on validation ({top:.4f}). Why: {reason}", ""]

    caveat = validation_caveat(task)
    if caveat:
        lines += ["> **Read the two columns above with care.** " + caveat, ""]

    if baseline and baseline["number"] != best["number"]:
        delta = best["metrics"][primary] - baseline["metrics"][primary]
        lines += [
            f"The starting model scored {baseline['metrics'][primary]:.4f} on validation and the best "
            f"submission scored {best['metrics'][primary]:.4f}, a change of {delta:+.4f}.",
            "",
        ]
        chance = _chance_over_baseline(project_dir, best)
        if chance is not None:
            lines += [f"Scored on the same resampled items, the best submission beats the starting "
                      f"model {chance:.0%} of the time: **{judge.verdict(chance)}**.", ""]
    elif baseline:
        lines += ["No submission beat the starting model, so the starting model is the result.", ""]

    found = _split_checks(task)
    if found:
        lines += ["## Raised when the task was set up", "",
                  "These were flagged about the data split before any modelling. Each affects how far the "
                  "score above can be trusted.", ""] + [f"- {item}" for item in found] + [""]

    lines += [
        "## What was tried",
        "",
        f"| # | What changed | {' | '.join(names)} |",
        "| --- | --- | " + " | ".join("---" for _ in names) + " |",
    ]
    lines += [_row(r, names) for r in records]
    usage = ["cd final", "python infer.py --model model --input my_data --output outputs.csv"
             + (" --n 100 --seed 0" if is_generation(task) else "")]
    lines += [
        "",
        f"Experiment {best['number']} is the result. Its exact code and model are in `final/`, with its "
        "outputs on the test set and the test labels beside them.",
        "",
        "## Budget",
        "",
        _budget_used(task, records),
        "",
        "## Using the model",
        "",
        "```bash",
        "mlguy serve                 # this project's page, once app.py has been built",
        *usage,
        "```",
        "",
    ]
    path = Path(project_dir) / "REPORT.md"
    path.write_text("\n".join(lines))
    return path

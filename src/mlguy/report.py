"""REPORT.md: the final scores, in the language of the brief."""

from __future__ import annotations

import json
from pathlib import Path

from mlguy import budget, judge
from mlguy.project import compare_with, state_dir, validation_caveat
from mlguy.records import Experiments


def _row(record: dict, primary: str) -> str:
    note = record.get("note", "").replace("|", "/")
    metrics = " | ".join(f"{v:.4f}" for v in record["metrics"].values()) if record["status"] == "ok" else "error"
    return f"| {record['number']} | {note} | {metrics} |"


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


def write(project_dir: Path, task: dict, records: list[dict], best: dict, test_result: dict) -> Path:
    primary = task["metric"]["primary"]
    names = list(best["metrics"].keys())
    baseline = next((r for r in records if r["number"] == 0 and r["status"] == "ok"), None)
    direction = "Higher is better" if task["task"].get("higher_is_better", True) else "Lower is better"
    output = task["task"]["output"].replace("_", " ")

    lines = [
        f"# {task['name']}: result",
        "",
        f"Predicts **{output}** from {task['task']['input'].replace('_', ' ')}. {direction}.",
        "",
        "## Score",
        "",
        f"Measured on {test_result['n']} held-out examples the model never saw during the search,",
        f"scored once after the budget was spent.",
        "",
        "| Metric | Test | Validation (best submission) |",
        "| --- | --- | --- |",
    ]
    for name in names:
        test_value = test_result["metrics"].get(name)
        lines.append(
            f"| {name} | {test_value:.4f} |" if test_value is None else
            f"| {name} | {test_value:.4f} | {best['metrics'][name]:.4f} |"
        )
    lines += ["", f"The score that counts is **{primary} = {test_result['metrics'][primary]:.4f}**.", ""]
    if test_result.get("interval"):
        low, high = test_result["interval"]
        lines += [f"On a different draw of {test_result['n']} test examples it would be expected to fall "
                  f"between {low:.4f} and {high:.4f} (95% interval).", ""]

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
            lines += [f"Scored on the same resampled validation rows, the best submission beats the starting "
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
    lines += [_row(r, primary) for r in records]
    lines += [
        "",
        f"Experiment {best['number']} is the result. Its exact code and model are kept in "
        f"`experiments/{best['number']:04d}/submission/`.",
        "",
        "## Budget",
        "",
        _budget_used(task, records),
        "",
        "## Using the model",
        "",
        "```bash",
        "mlguy serve                 # this project's page, once app.py has been built",
        f"python predict.py --model experiments/{best['number']:04d}/submission/model \\",
        "    --input my_data.csv --output predictions.csv",
        "```",
        "",
    ]
    path = Path(project_dir) / "REPORT.md"
    path.write_text("\n".join(lines))
    return path

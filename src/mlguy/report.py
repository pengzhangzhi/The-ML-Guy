"""REPORT.md: the final scores, in the language of the brief."""

from __future__ import annotations

from pathlib import Path

from mlguy import budget
from mlguy.project import validation_caveat


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
    elif baseline:
        lines += ["No submission beat the starting model, so the starting model is the result.", ""]

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

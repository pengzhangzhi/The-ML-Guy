"""REPORT.md: what was measured, as tables and plain statements of fact.

What the numbers mean for the user is not written here. That takes knowing their problem, so the
agent writes it at the top of this file afterwards (the deploy skill). What is here is the part
an agent should not be the one to choose: every metric, every submission, what was agreed.
"""

from __future__ import annotations

from pathlib import Path

from mlguy import budget
from mlguy.project import (direction, has_test_split, held_back_path, infer_args, state_dir,
                           validation_caveat)


MAX_COLUMNS = 4  # the experiment table shows the primary metric and a few others; log.jsonl has them all


def _all_metrics(primary: str, *results: dict) -> list[str]:
    """Every metric name any of these results reports, primary first.

    The score table shows all of them. The evaluator's author chose what to report, and a metric
    left out of the report is one the user does not see: one trial's evaluator reported nine, four
    reached REPORT.md, and the one that disagreed with the headline was among the five dropped.
    """
    names = [primary]
    for result in results:
        names += [name for name in result["metrics"] if name not in names]
    return names


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
    errors = sum(1 for r in records if r["status"] == "error")
    if errors:
        parts.append(f"{errors} submission(s) failed to run")
    line = (", ".join(parts).capitalize() + ".") if parts else "No submission limit was set."
    if budget.agreed_only(task):
        # Not checked by anyone but the agent, so it is reported as what was agreed, not as kept.
        line += f" Also agreed, and the agent's to keep: {budget.agreed_only(task)}."
    return line


def _audit(task: dict) -> str:
    skipped = state_dir(task) / "audit_skipped.txt"
    if skipped.exists():
        return ("> **The split and the evaluator were not audited before they were frozen.** "
                f"The reason given: {skipped.read_text().strip()}")
    return ("Before any modelling, a separate agent tried to break the split and the evaluator; "
            "what it found is in `task/audit.md`.")


def write(project_dir: Path, task: dict, records: list[dict], best: dict, test_result: dict,
          best_on_validation: dict | None = None, reason: str | None = None) -> Path:
    """`best` is the submission taken as the result; normally also the best on validation."""
    primary = task["metric"]["primary"]
    scored = _all_metrics(primary, best, test_result)   # the score table: all of them
    names = scored[:MAX_COLUMNS]                        # the experiment table: as many as fit
    better = "higher" if direction(task) == "maximize" else "lower"
    counted = sum(1 for r in records if r.get("counts_against_budget"))
    limit = task.get("budget", {}).get("max_submissions")
    used = f"{counted} of {limit} submissions" if limit else f"{counted} submissions"
    output = task["task"]["output"].replace("_", " ")
    given = task["task"].get("input", "").replace("_", " ")
    headline = f"Takes {given} and returns **{output}**." if given else f"Returns **{output}**."
    final_args = " ".join(infer_args(task, "test"))
    if has_test_split(task):
        measured = ["Measured on the held-out test split, which stayed outside the project until now,",
                    f"scored once, at the end of the search ({used} used)."]
        columns = f"| Metric | Test | Validation (experiment {best['number']}) |"
    else:
        # Nothing was held back, so the final check is a fresh run on validation. Say so, because
        # the two columns then share their inputs and are closer than a held-out test would be.
        measured = ["No test split was held back. The final check ran the model again on the validation",
                    "split" + (f", with {final_args}," if final_args else "")
                    + f" once, at the end of the search ({used} used)."]
        columns = f"| Metric | Final check | Validation (experiment {best['number']}) |"

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
    for name in scored:
        def cell(result: dict) -> str:
            value = result["metrics"].get(name)
            if value is None:
                return "refused" if result.get("refused") else "n/a"
            return f"{value:.4f}"
        lines.append(f"| {name} | {cell(test_result)} | {cell(best)} |")
    if test_result.get("refused"):
        # A refusal on the final check is the result, as a score would be. Saying so plainly matters
        # more than the validation column: that number was never confirmed on new data.
        lines += ["", "**The final check refused this model's outputs, so there is no test score.** "
                      f"The validation {primary} above was not confirmed on new data. The evaluator said:",
                  "", "```", test_result["refused"].strip(), "```", ""]
    else:
        lines += ["", f"The score that counts is **{primary} = {test_result['metrics'][primary]:.4f}** "
                      f"({better} is better).", ""]

    if best_on_validation and best_on_validation["number"] != best["number"]:
        top = best_on_validation["metrics"][primary]
        lines += [f"The result is experiment {best['number']} ({primary} {best['metrics'][primary]:.4f} on "
                  f"validation), chosen over experiment {best_on_validation['number']}, which scored best "
                  f"on validation ({top:.4f}). Why: {reason}", ""]

    caveat = validation_caveat(task)
    if caveat:
        lines += ["> **About the split, as agreed before the search:** " + caveat, ""]

    lines += [_audit(task), ""]

    lines += [
        "## What was tried",
        "",
        f"| # | What changed | {' | '.join(names)} |",
        "| --- | --- | " + " | ".join("---" for _ in names) + " |",
    ]
    lines += [_row(r, names) for r in records]
    if len(scored) > len(names):
        # Say which columns were left out, so a reader who only has this file knows there are more
        # and where they are. The score table above carries all of them for the result.
        lines += ["", f"This table shows {len(names)} of the evaluator's {len(scored)} metrics; it also "
                      f"reports {', '.join(scored[len(names):])}. Every submission's full set is in "
                      f"`experiments/log.jsonl`."]
    # The task fixes infer.py's extra arguments, so the example quotes them: run any other way,
    # the model is not the one that was scored.
    usage = ["cd final", " ".join(["python infer.py --model model --input my_data --output outputs",
                                   *infer_args(task, "validation")])]
    reference_kept = "test_reference" in task.get("data", {}) and held_back_path(task, "test_reference").exists()
    lines += [
        "",
        f"Experiment {best['number']} is the result. Its exact code and model are in `final/`, with its "
        "outputs on the test set beside them"
        + (", and the test reference, now that the score has been taken." if reference_kept else "."),
        "",
        "## Budget",
        "",
        _budget_used(task, records),
        "",
        "## Using the model",
        "",
        "```bash",
        *(["python app.py               # this project's page"] if (Path(project_dir) / "app.py").exists() else []),
        *usage,
        "```",
        "",
    ]
    cost = best.get("cost") or {}
    if cost:
        # What it costs to run, for whoever has to deploy it. Measured on this machine, so it is
        # an order of magnitude rather than a specification.
        measured = []
        if "model_mb" in cost:
            size = "under 1 MB" if cost["model_mb"] < 1 else f"{cost['model_mb']:.1f} MB"
            measured.append(f"The model is {size} on disk")
        if "predict_seconds" in cost:
            measured.append(f"it took {cost['predict_seconds']:.1f} s to run over the validation "
                            f"split on the machine it was scored on")
        lines += [" and ".join(measured) + ".", ""]
    path = Path(project_dir) / "REPORT.md"
    path.write_text("\n".join(lines))
    return path

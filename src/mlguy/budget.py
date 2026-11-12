"""The budget: how many scored submissions the search gets. `mlguy submit` refuses past it.

The count is the one limit that protects the score. Every look at validation fits the search to
it a little more, so how many looks there are is part of the rubric. Time, GPU-hours and money
are the user's limits too, but they do not change what a score means and the framework never
sees the compute. Whatever else `[budget]` holds is carried to the agent and the report as it
was written, and keeping it is the agent's job.
"""

from __future__ import annotations

ENFORCED = "max_submissions"


def state(task: dict, records: list[dict]) -> dict:
    """What the budget allows right now, and why it is spent if it is."""
    spec = task.get("budget", {})
    used = sum(1 for r in records if r.get("counts_against_budget", True))
    limits, reason = {}, None
    if ENFORCED in spec:
        limits["submissions"] = {"used": used, "limit": spec[ENFORCED]}
        if used >= spec[ENFORCED]:
            reason = f"all {spec[ENFORCED]} submissions are used"
    for key, value in spec.items():
        if key != ENFORCED:
            # TOML reads a bare date as a date, which `status --json` could not print.
            limits[key] = {"limit": value if isinstance(value, (int, float, str)) else str(value),
                           "enforced": False}
    return {"limits": limits, "spent": reason is not None, "reason": reason}


def agreed_only(task: dict) -> str | None:
    """The limits the user set that the framework does not enforce, as they were written."""
    rest = [f"{key} = {value}" for key, value in task.get("budget", {}).items() if key != ENFORCED]
    return ", ".join(rest) or None


def describe(task: dict, records: list[dict]) -> str:
    """One line for a human or an agent."""
    status = state(task, records)
    if not status["limits"]:
        return "No budget set."
    if status["spent"]:
        # "Run `mlguy finish`" after it has already run sends an agent back to a command that will
        # refuse it, and reads as though the project were unfinished. The final score is taken once,
        # so once it exists there is nothing left to run.
        from mlguy.project import state_dir   # here, so this module stays a leaf at import time
        if (state_dir(task) / "test_score.json").exists():
            return f"Budget spent: {status['reason']}. The final score has been taken; see REPORT.md."
        return f"Budget spent: {status['reason']}. Run `mlguy finish`."
    parts = []
    if "submissions" in status["limits"]:
        s = status["limits"]["submissions"]
        parts.append(f"Budget: {s['limit'] - s['used']} of {s['limit']} submissions left.")
    if agreed_only(task):
        parts.append(f"Also agreed with the user: {agreed_only(task)}. `mlguy` does not enforce "
                     "these: keep them yourself.")
    return " ".join(parts)

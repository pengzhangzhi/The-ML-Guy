"""The budget: how long the search may run, and optionally how many submissions it gets.

Time is the limit that is enforced by default. `hours` counts from `mlguy init`, and once it is
up `mlguy submit` refuses; `mlguy finish` still runs. The number of submissions is recorded and
reported rather than capped: the validation reference sits inside the project and `mlguy score`
reads it for free, so a cap would not stop the search looking at validation, only stop it
recording what it looked at. A user who wants a cap anyway sets `max_submissions`, and that is
enforced too. Whatever else `[budget]` holds is carried to the agent and the report as it was
written, and keeping it is the agent's job: the framework never sees the compute.
"""

from __future__ import annotations

from datetime import datetime, timezone

ENFORCED = ("hours", "max_submissions")
STARTED = "search_started.txt"


def started(task: dict) -> datetime | None:
    """When the search began: the moment `mlguy init` froze the task."""
    from mlguy.project import state_dir   # here, so this module stays a leaf at import time
    path = state_dir(task) / STARTED
    return datetime.fromisoformat(path.read_text().strip()) if path.exists() else None


def hours_used(task: dict) -> float | None:
    """From the freeze to now, or to the final check once it has been taken."""
    import json
    from mlguy.project import state_dir
    start = started(task)
    if start is None:
        return None
    final = state_dir(task) / "test_score.json"
    end = (datetime.fromisoformat(json.loads(final.read_text())["time"]) if final.exists()
           else datetime.now(timezone.utc))
    return (end - start).total_seconds() / 3600


def state(task: dict, records: list[dict]) -> dict:
    """What the budget allows right now, and why it is spent if it is."""
    spec = task.get("budget", {})
    used = sum(1 for r in records if r.get("counts_against_budget", True))
    limits, reason = {}, None
    limits["submissions"] = {"used": used}
    if "max_submissions" in spec:
        limits["submissions"]["limit"] = spec["max_submissions"]
        if used >= spec["max_submissions"]:
            reason = f"all {spec['max_submissions']} submissions are used"
    elapsed = hours_used(task)
    if elapsed is not None or "hours" in spec:
        limits["hours"] = {"used": round(elapsed or 0.0, 2)}
        if "hours" in spec:
            limits["hours"]["limit"] = spec["hours"]
            if (elapsed or 0.0) >= spec["hours"]:
                reason = f"the {_h(spec['hours'])} for the search are up"
    for key, value in spec.items():
        if key not in ENFORCED:
            # TOML reads a bare date as a date, which `status --json` could not print.
            limits[key] = {"limit": value if isinstance(value, (int, float, str)) else str(value),
                           "enforced": False}
    return {"limits": limits, "spent": reason is not None, "reason": reason}


def agreed_only(task: dict) -> str | None:
    """The limits the user set that the framework does not enforce, as they were written."""
    rest = [f"{key} = {value}" for key, value in task.get("budget", {}).items() if key not in ENFORCED]
    return ", ".join(rest) or None


def _h(hours: float) -> str:
    return f"{hours:.1f} h" if hours < 10 else f"{hours:.0f} h"


def describe(task: dict, records: list[dict]) -> str:
    """One line for a human or an agent."""
    status = state(task, records)
    if status["spent"]:
        # "Run `mlguy finish`" after it has already run sends an agent back to a command that will
        # refuse it, and reads as though the project were unfinished. The final score is taken once,
        # so once it exists there is nothing left to run.
        from mlguy.project import state_dir
        if (state_dir(task) / "test_score.json").exists():
            return f"Budget spent: {status['reason']}. The final score has been taken; see REPORT.md."
        return f"Budget spent: {status['reason']}. Run `mlguy finish`."
    parts = []
    hours = status["limits"].get("hours")
    if hours and "limit" in hours:
        left = hours["limit"] - hours["used"]
        parts.append(f"Time: {_h(hours['used'])} of {_h(hours['limit'])} used, {_h(left)} left.")
    elif hours:
        parts.append(f"Time: {_h(hours['used'])} since the task was frozen; no limit set.")
    s = status["limits"]["submissions"]
    if "limit" in s:
        parts.append(f"Submissions: {s['limit'] - s['used']} of {s['limit']} left.")
    else:
        parts.append(f"Submissions: {s['used']} so far, no limit.")
    if agreed_only(task):
        parts.append(f"Also agreed with the user: {agreed_only(task)}. `mlguy` does not enforce "
                     "these: keep them yourself.")
    return " ".join(parts)

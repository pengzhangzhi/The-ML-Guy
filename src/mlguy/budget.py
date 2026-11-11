"""The budget: whatever limit fits the user. `mlguy submit` refuses once any one is reached.

Enforced now: number of submissions, wall-clock hours from the first submission, a deadline.
Recorded but not enforced: GPU-hours and money, which need the Slurm backend.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

ENFORCED = ("max_submissions", "max_hours", "deadline")
RECORDED_ONLY = ("max_gpu_hours", "max_usd")


def _parse(stamp: str) -> datetime:
    value = datetime.fromisoformat(stamp)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def state(task: dict, records: list[dict]) -> dict:
    """What the budget allows right now, and why it is spent if it is."""
    spec = task.get("budget", {})
    counted = [r for r in records if r.get("counts_against_budget", True)]
    used = len(counted)
    started = _parse(counted[0]["time"]) if counted else None
    now = datetime.now(timezone.utc)

    limits, spent = {}, []
    if "max_submissions" in spec:
        limits["submissions"] = {"used": used, "limit": spec["max_submissions"]}
        if used >= spec["max_submissions"]:
            spent.append(f"all {spec['max_submissions']} submissions are used")
    if "max_hours" in spec:
        hours = (now - started).total_seconds() / 3600 if started else 0.0
        limits["hours"] = {"used": round(hours, 2), "limit": spec["max_hours"]}
        if started and hours >= spec["max_hours"]:
            spent.append(f"{spec['max_hours']} hours have passed since the first submission")
    if "deadline" in spec:
        deadline = _parse(str(spec["deadline"]))
        limits["deadline"] = {"at": deadline.isoformat(timespec="seconds")}
        if now >= deadline:
            spent.append(f"the deadline of {deadline.isoformat(timespec='seconds')} has passed")
    for key in RECORDED_ONLY:
        if key in spec:
            limits[key] = {"limit": spec[key], "enforced": False}

    return {"limits": limits, "spent": bool(spent), "reason": "; ".join(spent) or None}


def describe(task: dict, records: list[dict]) -> str:
    """One line for a human or an agent."""
    status = state(task, records)
    if not status["limits"]:
        return "No budget set."
    if status["spent"]:
        return f"Budget spent: {status['reason']}. Run `mlguy finish`."
    parts = []
    if "submissions" in status["limits"]:
        s = status["limits"]["submissions"]
        parts.append(f"{s['limit'] - s['used']} of {s['limit']} submissions left")
    if "hours" in status["limits"]:
        h = status["limits"]["hours"]
        parts.append(f"{max(0.0, h['limit'] - h['used']):.2f} of {h['limit']} hours left")
    if "deadline" in status["limits"]:
        remaining = _parse(status["limits"]["deadline"]["at"]) - datetime.now(timezone.utc)
        parts.append(f"deadline in {max(timedelta(0), remaining)}")
    return "Budget: " + ", ".join(parts) + "."

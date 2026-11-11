"""The experiment dashboard. The same page for every project; it reads experiments/log.jsonl.

    mlguy dashboard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from mlguy import budget
from mlguy.judge import METRICS
from mlguy.project import load_task, state_dir
from mlguy.records import Experiments

REFRESH_MS = 5000


def main() -> None:
    project_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    st.set_page_config(page_title="mlguy experiments", layout="wide")

    try:
        task = load_task(project_dir)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    experiments = Experiments(project_dir)
    records = experiments.all()
    primary = task["metric"]["primary"]
    maximise = METRICS[primary][1] == "maximize"
    best = experiments.best(primary)

    st.title(task["name"])
    st.caption(
        f"Predicting {task['task']['output'].replace('_', ' ')} from a "
        f"{task['task']['input'].replace('_', ' ')} · hill-climbing on validation {primary}"
    )

    if not records:
        st.info("No experiments yet. Train a model into `model/`, then run `mlguy validate`.")
        return

    scored = [r for r in records if r["status"] == "ok"]
    baseline = next((r for r in scored if r["number"] == 0), None)
    final_path = state_dir(task) / "test_score.json"
    final = json.loads(final_path.read_text()) if final_path.exists() else None

    columns = st.columns(4)
    columns[0].metric(
        f"best validation {primary}",
        f"{best['metrics'][primary]:.4f}" if best else "-",
        delta=f"{best['metrics'][primary] - baseline['metrics'][primary]:+.4f} vs baseline"
        if best and baseline and best["number"] != 0 else None,
    )
    columns[1].metric("experiments", f"{len(records)}", delta=f"{len(records) - len(scored)} errors" or None)
    columns[2].metric(
        f"test {primary}", f"{final['metrics'][primary]:.4f}" if final else "not scored yet",
        help="Scored once, after the budget is spent.",
    )
    status = budget.state(task, records)
    columns[3].metric("budget", "spent" if status["spent"] else "open")
    st.caption(budget.describe(task, records))

    st.subheader("Best so far")
    if scored:
        series = []
        best_value = None
        for record in scored:
            value = record["metrics"][primary]
            if best_value is None or (value > best_value if maximise else value < best_value):
                best_value = value
            series.append({"experiment": record["number"], "this experiment": value, "best so far": best_value})
        chart = pd.DataFrame(series).set_index("experiment")
        st.line_chart(chart, height=320)

    st.subheader("What was tried")
    rows = []
    for record in records:
        row = {"#": record["number"], "what changed": record.get("note", ""), "status": record["status"]}
        row.update({k: round(v, 4) for k, v in record.get("metrics", {}).items()})
        if record["status"] == "error":
            row["error"] = record["error"].splitlines()[0][:80]
        row["best"] = "yes" if best and record["number"] == best["number"] else ""
        row["time"] = record["time"]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows[::-1]), width="stretch", hide_index=True)

    st.subheader("One experiment")
    numbers = [r["number"] for r in records][::-1]
    chosen = st.selectbox("experiment", numbers, format_func=lambda n: f"experiment {n}")
    record = next(r for r in records if r["number"] == chosen)
    left, right = st.columns([1, 2])
    left.json(record)
    folder = experiments.dir / f"{chosen:04d}" / "submission"
    if folder.exists():
        files = sorted(p for p in folder.rglob("*.py"))
        if files:
            pick = right.selectbox("code in this submission", files, format_func=lambda p: str(p.relative_to(folder)))
            right.code(pick.read_text(), language="python")
    else:
        right.info("no saved submission for this experiment")

    st.caption(f"reading {experiments.log}")
    st.button("refresh")


main()

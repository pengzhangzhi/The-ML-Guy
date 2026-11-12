"""Shared by the tests: write full tables out in the layout a task uses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_TOML = '''\
train = "data/train.csv"
validation_inputs = "data/validation_inputs.csv"
validation_labels = "data/validation_labels.csv"
test_inputs = "data/test_inputs.csv"
test_labels = "data/test_labels.csv"
'''


def write_splits(data_dir: Path, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
                 target: str = "target") -> None:
    """Give every example an id, and split validation and test into inputs and labels."""
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / "train.csv", index=False)
    for name, part in (("validation", validation), ("test", test)):
        part = part.reset_index(drop=True)
        part.insert(0, "id", [f"{name[0]}{i}" for i in range(len(part))])
        part.drop(columns=[target]).to_csv(data_dir / f"{name}_inputs.csv", index=False)
        part[["id", target]].to_csv(data_dir / f"{name}_labels.csv", index=False)


# The tests' stand-in for a task's evaluator: matches outputs to labels by id and reports
# accuracy, and Spearman and Pearson when both sides are numbers.
EVALUATE_PY = '''\
import argparse, json, math
import pandas as pd
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("--outputs"); p.add_argument("--reference"); p.add_argument("--inputs")
p.add_argument("--split"); p.add_argument("--result")
a = p.parse_args()
out = pd.read_csv(a.outputs, dtype={"id": str})
ref = pd.read_csv(a.reference, dtype={"id": str})
if "prediction" not in out or out["id"].duplicated().any() or set(out["id"]) != set(ref["id"]):
    raise SystemExit("need one prediction for every id")
both = ref.merge(out, on="id")
metrics = {"accuracy": float((both["prediction"].astype(str) == both["target"].astype(str)).mean())}
for name, f in (("spearman", stats.spearmanr), ("pearson", stats.pearsonr)):
    try:
        metrics[name] = float(f(both["target"].astype(float), both["prediction"].astype(float))[0])
    except (ValueError, TypeError):
        metrics[name] = math.nan
if not math.isfinite(metrics.get("spearman", math.nan)):
    metrics["spearman"] = math.nan
json.dump({"metrics": metrics}, open(a.result, "w"))
'''

APPROVED_BRIEF = "# Brief\n\nApproved by the user: yes, go ahead (2026-11-11)\n"


def write_task(project: Path, toml: str) -> None:
    """task.toml, an approved brief, an audit, and the stand-in evaluator (a test may replace it)."""
    task = project / "task"
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(toml)
    (task / "brief.md").write_text(APPROVED_BRIEF)
    (task / "audit.md").write_text("# Audit\n\nNothing found that should stop the freeze.\n")
    (task / "evaluate.py").write_text(EVALUATE_PY)

"""A worked case: one output per example, matched by id to a known answer in the reference.

This is `reference/evaluate.py` filled in for that shape of problem — a CSV of `id, prediction`
scored against a reference CSV of `id, target`. Use it if your problem has that shape; the id
matching and the refusals are the part worth keeping. `score` is still yours.

    python evaluate.py --outputs FILE --reference FILE [--inputs PATH] \
                       --split validation|final --result FILE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def score(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """>>> REPLACE THIS. Everything above and below is the contract. <<<

    Return every number worth reporting, with task.toml's `[metric] primary` among them and
    finite. Report the ones that disagree with your headline too: a metric a report leaves out
    is one the user never sees, and the one that disagrees is the one worth seeing.
    """
    return {
        "spearman": float(stats.spearmanr(y, prediction).statistic),   # the primary, say
        "mae": float(np.abs(y - prediction).mean()),                   # and whatever else helps
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--inputs")          # accept it even when this metric ignores it
    parser.add_argument("--split", choices=["validation", "final"], required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    outputs = pd.read_csv(args.outputs, dtype={"id": str})
    reference = pd.read_csv(args.reference, dtype={"id": str})

    # Refuse what cannot be scored, with an error. Never score it as zero: a zero is a result the
    # search will try to beat, and a model that crashes is not a model that scored badly.
    if "prediction" not in outputs.columns:
        raise SystemExit("outputs need a `prediction` column")
    if outputs["id"].duplicated().any():
        raise SystemExit("an id appears twice in the outputs")
    missing = set(reference["id"]) - set(outputs["id"])
    unknown = set(outputs["id"]) - set(reference["id"])
    if missing or unknown:
        raise SystemExit(f"{len(missing)} ids missing from the outputs, {len(unknown)} not in this split")

    both = reference.merge(outputs, on="id")          # matched by id, so row order cannot matter
    y = both["target"].to_numpy(float)
    prediction = both["prediction"].to_numpy(float)
    if not np.isfinite(prediction).all():
        raise SystemExit("some predictions are NaN or infinite")

    metrics = score(y, prediction)

    # --result is the only file anything reads back, and this script runs with task/ as its
    # working directory, so a relative path written anywhere else lands in the frozen task and
    # breaks its fingerprint. Write this one file and nothing else.
    Path(args.result).write_text(json.dumps({"metrics": metrics}))


if __name__ == "__main__":
    main()

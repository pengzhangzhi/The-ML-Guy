"""A worked evaluator for a generation task. Copy it, replace the marked part, delete the rest.

A generative task has no answer key, so this script is the entire definition of what good means.
The contract below is what stops it being gamed; `quality` is yours.

    python evaluate.py --outputs FILE --reference CONDITIONS --inputs PATH \
                       --split validation|final --result FILE

--reference lists the conditions to generate for, by id, and carries no target. --outputs holds
`n` rows for every one of them, each row carrying its condition's id.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd


def quality(sample: str, condition) -> float:
    """>>> REPLACE THIS. Everything around it is the contract. <<<

    How good is ONE sample for ONE condition, on its own. Whatever the user said they would use
    to judge a batch by hand: a folding model, a classifier, a simulation, a comparison with real
    data. `condition` is this id's row from --inputs, so the score can depend on what was asked for.
    """
    raise NotImplementedError("score one sample against its condition")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--inputs", required=True)   # generation almost always needs the conditions
    parser.add_argument("--split", choices=["validation", "final"], required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    # How many samples each condition gets is fixed by the task, so read it from task.toml rather
    # than hard-coding it. Resolved against this file, not the working directory.
    task = tomllib.loads((Path(__file__).parent / "task.toml").read_text())
    expected = task["sampling"]["final_n" if args.split == "final" else "n"]

    outputs = pd.read_csv(args.outputs, dtype={"id": str})
    conditions = pd.read_csv(args.inputs, dtype={"id": str}).set_index("id")
    wanted = pd.read_csv(args.reference, dtype={"id": str})["id"].tolist()

    if "sample" not in outputs.columns:
        raise SystemExit("outputs need a `sample` column")
    unknown = set(outputs["id"]) - set(wanted)
    if unknown:
        raise SystemExit(f"{len(unknown)} ids in the outputs are not conditions of this split")

    per_condition = outputs.groupby("id")["sample"].apply(list)
    for condition_id in wanted:
        samples = per_condition.get(condition_id, [])
        # The count is load-bearing, not bookkeeping: the best of twenty samples scores far better
        # than twenty do, so a short batch is a different measurement, not a smaller one.
        if len(samples) != expected:
            raise SystemExit(f"{condition_id} has {len(samples)} samples, not {expected}")
        if len(set(samples)) != len(samples):
            raise SystemExit(f"{condition_id} repeats a sample; the task asks for {expected} distinct ones")

    # Score EVERY sample and average. Taking the maximum over a condition's samples rewards
    # sampling more rather than generating better, and the model cannot tell which one you kept.
    scores = np.array([quality(sample, conditions.loc[condition_id])
                       for condition_id in wanted
                       for sample in per_condition[condition_id]], dtype=float)
    by_condition = scores.reshape(len(wanted), expected)

    metrics = {
        "mean_quality": float(scores.mean()),             # the primary, say
        "worst_condition": float(by_condition.mean(1).min()),   # one bad condition should show
        "conditions": float(len(wanted)),
    }
    Path(args.result).write_text(json.dumps({"metrics": metrics}))


if __name__ == "__main__":
    main()

"""A worked case: the model draws samples for each condition, judged without an answer key.

This is `reference/evaluate.py` filled in for that shape of problem. --inputs is a CSV of the
conditions to generate for, each with an `id`; infer.py is called with `[infer] args` such as
`--n 20 --seed 0` and writes `n` rows per condition (`id, sample`). There is no answer to match,
so this script is the whole definition of good. `quality` is yours.

    python evaluate.py --outputs FILE --inputs CONDITIONS [--reference PATH] \
                       --split validation|final --result FILE
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
    parser.add_argument("--reference")               # real examples to compare against, if any
    parser.add_argument("--inputs", required=True)   # the conditions
    parser.add_argument("--split", choices=["validation", "final"], required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    # How many samples each condition gets is fixed by the task's [infer] args, so read it from
    # task.toml rather than hard-coding it. Resolved against this file, not the working directory.
    infer = tomllib.loads((Path(__file__).parent / "task.toml").read_text())["infer"]
    expected = int((infer.get("final_args", infer["args"]) if args.split == "final" else infer["args"])["n"])

    outputs = pd.read_csv(args.outputs, dtype={"id": str})
    conditions = pd.read_csv(args.inputs, dtype={"id": str}).set_index("id")
    wanted = conditions.index.tolist()

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

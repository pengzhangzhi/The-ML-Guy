"""Scoring for the amylase project: how well the predicted activity ranks the variants.

    python evaluate.py --outputs FILE --reference LABELS --split validation|final --result FILE

spearman (primary): rank correlation between predicted and measured activity.
ndcg: how much of the best activity sits at the top of the predicted ranking; linear gain,
      with measured values shifted to start at zero.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("--outputs", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--split")
p.add_argument("--result", required=True)
a = p.parse_args()

out = pd.read_csv(a.outputs, dtype={"id": str})
ref = pd.read_csv(a.reference, dtype={"id": str})
if "prediction" not in out or out["id"].duplicated().any() or set(out["id"]) != set(ref["id"]):
    raise SystemExit("the outputs need exactly one numeric `prediction` for every id")
both = ref.merge(out, on="id")
y, pred = both["target"].to_numpy(float), both["prediction"].to_numpy(float)
if np.ptp(pred) == 0:
    raise SystemExit("all predictions are identical, so the ranking is undefined")

rel = y - y.min()
discount = 1.0 / np.log2(np.arange(2, len(rel) + 2))
ideal = float((np.sort(rel)[::-1] * discount).sum())
ndcg = float((rel[np.argsort(-pred, kind="stable")] * discount).sum() / ideal)

json.dump({"metrics": {"spearman": float(stats.spearmanr(y, pred).statistic), "ndcg": ndcg}}, open(a.result, "w"))

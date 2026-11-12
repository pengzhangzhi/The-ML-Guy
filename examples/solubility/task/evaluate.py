"""Scoring for the solubility project: soluble (1) or not (0), with the probability behind it.

    python evaluate.py --outputs FILE --reference LABELS --split validation|final --result FILE

accuracy (primary, what the DeepSol and PEER papers report), balanced_accuracy, mcc, and
roc_auc from the `score` column (the probability of being soluble).
"""

import argparse
import json

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score

p = argparse.ArgumentParser()
p.add_argument("--outputs", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--inputs"); p.add_argument("--split")
p.add_argument("--result", required=True)
a = p.parse_args()

out = pd.read_csv(a.outputs, dtype={"id": str})
ref = pd.read_csv(a.reference, dtype={"id": str})
if not {"prediction", "score"} <= set(out) or out["id"].duplicated().any() or set(out["id"]) != set(ref["id"]):
    raise SystemExit("the outputs need exactly one `prediction` (0 or 1) and `score` for every id")
both = ref.merge(out, on="id")
if not set(both["prediction"].unique()) <= {0, 1}:
    raise SystemExit("`prediction` must be the class, 0 or 1; put the probability in `score`")
y, pred = both["target"].astype(int), both["prediction"].astype(int)
json.dump({"metrics": {
    "accuracy": accuracy_score(y, pred),
    "balanced_accuracy": balanced_accuracy_score(y, pred),
    "mcc": matthews_corrcoef(y, pred),
    "roc_auc": roc_auc_score(y, both["score"].astype(float)),
}}, open(a.result, "w"))

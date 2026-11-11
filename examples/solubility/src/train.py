"""Fit a solubility model and write it to model/.

    python src/train.py --approach composition      # the baseline
    python src/train.py --approach dipeptide
    python src/train.py --approach dipeptide-gbt

Hyperparameters are chosen on the validation set here, so that `mlguy submit` is an
independent score rather than a repeat of the tuning.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mlguy.judge import score

import features

STRENGTHS = [0.01, 0.1, 1.0, 10.0]
ROOT = Path(__file__).resolve().parents[1]


def evaluate(model, x, y, metrics: list[str]) -> dict[str, float]:
    positive = max(model.classes_)
    y_score = model.predict_proba(x)[:, list(model.classes_).index(positive)]
    return score(y, model.predict(x), metrics, y_score=y_score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", default="composition",
                        choices=["composition", "dipeptide", "dipeptide-gbt", "dipeptide-gbt-tuned"])
    parser.add_argument("--task", type=Path, default=ROOT / "task")
    parser.add_argument("--out", type=Path, default=ROOT / "model")
    args = parser.parse_args()

    with open(args.task / "task.toml", "rb") as f:
        task = tomllib.load(f)
    seq, tgt = "sequence", "target"
    alphabet = task["input"]["alphabet"]
    primary = task["metric"]["primary"]
    report = [primary] + [m for m in task["metric"]["secondary"] if m != primary]

    train = pd.read_csv(args.task / task["data"]["train"])
    val = pd.read_csv(args.task / task["data"]["validation_inputs"]).merge(
        pd.read_csv(args.task / task["data"]["validation_labels"]), on="id")
    y_train, y_val = train[tgt].to_numpy(), val[tgt].to_numpy()

    kind = "composition" if args.approach == "composition" else "dipeptide"
    bundle = {"alphabet": alphabet, "features": kind, "classification": True}
    x_train = features.build(train[seq].tolist(), bundle)
    x_val = features.build(val[seq].tolist(), bundle)
    print(f"{kind} features: {x_train.shape[1]} columns")

    if args.approach.startswith("dipeptide-gbt"):
        bundle["scaler"] = None
        settings = [{"max_iter": 400, "learning_rate": 0.08, "max_leaf_nodes": 31}]
        if args.approach == "dipeptide-gbt-tuned":
            settings = [
                {"max_iter": 1200, "learning_rate": 0.05, "max_leaf_nodes": 63},
                {"max_iter": 1200, "learning_rate": 0.05, "max_leaf_nodes": 127, "min_samples_leaf": 40},
                {"max_iter": 2000, "learning_rate": 0.03, "max_leaf_nodes": 63, "min_samples_leaf": 40},
            ]
        best = None
        for setting in settings:
            model = HistGradientBoostingClassifier(
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                random_state=0, **setting,
            ).fit(x_train, y_train)
            metrics = evaluate(model, x_val, y_val, report)
            print(f"  {setting} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if best is None or metrics[primary] > best[1][primary]:
                best = (model, metrics)
    else:
        scaler = StandardScaler().fit(x_train)
        bundle["scaler"] = scaler
        x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)
        best = None
        for strength in STRENGTHS:
            model = LogisticRegression(C=strength, max_iter=3000).fit(x_train, y_train)
            metrics = evaluate(model, x_val, y_val, report)
            print(f"  C={strength:<6} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if best is None or metrics[primary] > best[1][primary]:
                best = (model, metrics)

    model, metrics = best
    bundle["model"] = model
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out / "bundle.joblib")
    print(f"saved {args.approach}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()

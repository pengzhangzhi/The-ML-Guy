"""Baseline: a linear model on simple features, tuned on the validation set.

It exists to prove the pipeline runs end to end and to give later work something to beat.
Replace it with whatever you think will do better: this file and everything in src/ is yours.

    python src/train.py
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from mlguy.judge import METRICS, score

import features

STRENGTHS = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, default=ROOT / "task")
    parser.add_argument("--out", type=Path, default=ROOT / "model")
    args = parser.parse_args()

    with open(args.task / "task.toml", "rb") as f:
        task = tomllib.load(f)
    columns, target = list(task["data"]["input_columns"]), task["data"]["target_column"]
    is_classification = task["task"]["type"] == "classification"
    primary = task["metric"]["primary"]

    train = pd.read_csv(args.task / task["data"]["train"])
    val = pd.read_csv(args.task / task["data"]["validation"])
    y_train, y_val = train[target].to_numpy(), val[target].to_numpy()

    spec = features.describe(train, columns, task.get("input"))
    print(f"encoding: {features.summarise(spec)}")
    x_train, x_val = features.build(train, spec), features.build(val, spec)
    scaler = StandardScaler().fit(x_train)
    x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)
    print(f"{x_train.shape[1]} columns, {len(x_train)} training rows")

    best = None
    for strength in STRENGTHS:
        if is_classification:
            model = LogisticRegression(C=strength, max_iter=2000).fit(x_train, y_train)
            positive = max(model.classes_)
            y_score = model.predict_proba(x_val)[:, list(model.classes_).index(positive)]
            value = score(y_val, model.predict(x_val), [primary], y_score=y_score)[primary]
        else:
            model = Ridge(alpha=strength).fit(x_train, y_train)
            value = score(y_val, model.predict(x_val), [primary])[primary]
        print(f"  strength={strength:<7} validation {primary}={value:.4f}")
        better = best is None or (value > best[1] if METRICS[primary][1] == "maximize" else value < best[1])
        if better:
            best = (model, value, strength)

    model, value, strength = best
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"spec": spec, "scaler": scaler, "model": model,
                 "classification": is_classification}, args.out / "bundle.joblib")
    print(f"saved {'logistic' if is_classification else 'ridge'} model "
          f"(strength={strength}, validation {primary}={value:.4f}) to {args.out}")


if __name__ == "__main__":
    main()

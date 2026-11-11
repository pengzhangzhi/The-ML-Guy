"""Baseline: a linear model on simple features, tuned on the validation set.

It exists to prove the pipeline runs end to end and to give later work something to beat.
Replace it with whatever you think will do better.

    python src/train.py

Which features: one-hot per position if every sequence has the same length (variants of one
protein), otherwise residue composition (unrelated proteins of different lengths).
Which model: ridge regression, or logistic regression if the task is classification.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import joblib
import numpy as np
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
    seq, tgt = task["data"]["sequence_column"], task["data"]["target_column"]
    alphabet = task["input"]["alphabet"]
    length = task["input"].get("length")
    is_classification = task["task"]["type"] == "classification"
    primary = task["metric"]["primary"]

    train = pd.read_csv(args.task / task["data"]["train"])
    val = pd.read_csv(args.task / task["data"]["validation"])
    y_train, y_val = train[tgt].to_numpy(), val[tgt].to_numpy()

    bundle = {"alphabet": alphabet}
    if length:
        bundle.update(features="onehot", length=length)
    else:
        bundle["features"] = "composition"
    x_train = features.build(train[seq].tolist(), bundle)
    x_val = features.build(val[seq].tolist(), bundle)
    print(f"{bundle['features']} features: {x_train.shape[1]} columns, {len(x_train)} training rows")

    scaler = StandardScaler().fit(x_train) if bundle["features"] == "composition" else None
    if scaler is not None:
        x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)

    best = None
    for strength in STRENGTHS:
        if is_classification:
            model = LogisticRegression(C=strength, max_iter=2000).fit(x_train, y_train)
            y_score = model.predict_proba(x_val)[:, list(model.classes_).index(max(model.classes_))]
            value = score(y_val, model.predict(x_val), [primary], y_score=y_score)[primary]
        else:
            model = Ridge(alpha=strength).fit(x_train, y_train)
            value = score(y_val, model.predict(x_val), [primary])[primary]
        print(f"  strength={strength:<7} validation {primary}={value:.4f}")
        better = best is None or (value > best[1] if METRICS[primary][1] == "maximize" else value < best[1])
        if better:
            best = (model, value, strength)

    model, value, strength = best
    bundle.update(model=model, scaler=scaler, strength=strength, classification=is_classification)
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out / "bundle.joblib")
    print(f"saved {'logistic' if is_classification else 'ridge'} model "
          f"(strength={strength}, validation {primary}={value:.4f}) to {args.out}")


if __name__ == "__main__":
    main()

"""Baseline: one-hot encoding and ridge regression, with the ridge strength picked on
validation. It exists to prove the pipeline runs end to end and to give later work something
to beat. Replace it with whatever you think will do better.

    python src/train.py
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

import features

ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
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

    train = pd.read_csv(args.task / task["data"]["train"])
    val = pd.read_csv(args.task / task["data"]["validation"])
    length = task["input"].get("length") or len(train[seq].iloc[0])

    x_train, y_train = features.one_hot(train[seq].tolist(), alphabet, length), train[tgt].to_numpy()
    x_val, y_val = features.one_hot(val[seq].tolist(), alphabet, length), val[tgt].to_numpy()

    best = None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(x_train, y_train)
        rho = float(spearmanr(y_val, model.predict(x_val)).statistic)
        print(f"  alpha={alpha:<7} validation spearman={rho:.4f}")
        if best is None or rho > best[1]:
            best = (model, rho, alpha)

    model, rho, alpha = best
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": "onehot", "alphabet": alphabet, "length": length,
                 "alpha": alpha, "model": model}, args.out / "bundle.joblib")
    print(f"saved one-hot ridge (alpha={alpha}, validation spearman {rho:.4f}) to {args.out}")


if __name__ == "__main__":
    main()

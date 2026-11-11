"""Fit a model and write it to model/.

    python src/train.py --approach onehot-ridge     # the baseline
    python src/train.py --approach mutation-ridge
    python src/train.py --approach mutation-gbt
    python src/train.py --approach ensemble

Every approach is scored on the validation set here so the choice of hyperparameters is made
before `mlguy submit` is called; submit is the independent score.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import features

ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
ROOT = Path(__file__).resolve().parents[1]


def load_data(task_dir: Path, task: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data, seq, tgt = task["data"], task["data"]["input_columns"][0], task["data"]["target_column"]
    return (pd.read_csv(task_dir / data["train"])[[seq, tgt]],
            pd.read_csv(task_dir / data["validation"])[[seq, tgt]])


def fit_ridge(x_train, y_train, x_val, y_val) -> tuple[Ridge, float, float]:
    best = None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(x_train, y_train)
        rho = float(spearmanr(y_val, model.predict(x_val)).statistic)
        print(f"  alpha={alpha:<7} validation spearman={rho:.4f}")
        if best is None or rho > best[1]:
            best = (model, rho, alpha)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", default="onehot-ridge",
                        choices=["onehot-ridge", "mutation-ridge", "mutation-gbt", "ensemble"])
    parser.add_argument("--task", type=Path, default=ROOT / "task")
    parser.add_argument("--out", type=Path, default=ROOT / "model")
    args = parser.parse_args()

    with open(args.task / "task.toml", "rb") as f:
        task = tomllib.load(f)
    seq, tgt = task["data"]["input_columns"][0], task["data"]["target_column"]
    alphabet, length = task["input"]["alphabet"], task["input"]["length"]
    reference = task["input"]["reference"]

    train, val = load_data(args.task, task)
    s_train, y_train = train[seq].tolist(), train[tgt].to_numpy()
    s_val, y_val = val[seq].tolist(), val[tgt].to_numpy()

    bundle = {"alphabet": alphabet, "length": length, "reference": reference, "approach": args.approach}

    if args.approach == "onehot-ridge":
        bundle["features"] = "onehot"
        x_train, x_val = features.one_hot(s_train, alphabet, length), features.one_hot(s_val, alphabet, length)
        model, rho, alpha = fit_ridge(x_train, y_train, x_val, y_val)
        bundle.update(model=model, alpha=alpha)

    elif args.approach in ("mutation-ridge", "mutation-gbt"):
        vocab = features.mutation_vocab(s_train, reference)
        bundle["features"], bundle["vocab"] = "mutations", vocab
        x_train = features.mutation_matrix(s_train, reference, vocab)
        x_val = features.mutation_matrix(s_val, reference, vocab)
        print(f"  {len(vocab)} substitutions seen in training, {x_train.shape[1]} features")
        if args.approach == "mutation-ridge":
            model, rho, alpha = fit_ridge(x_train, y_train, x_val, y_val)
            bundle.update(model=model, alpha=alpha)
        else:
            model = HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=0,
            ).fit(x_train, y_train)
            rho = float(spearmanr(y_val, model.predict(x_val)).statistic)
            print(f"  validation spearman={rho:.4f}")
            bundle.update(model=model)

    else:  # ensemble: average the ranks of the two ridge models
        vocab = features.mutation_vocab(s_train, reference)
        parts = []
        for kind, x_tr, x_va in [
            ("onehot", features.one_hot(s_train, alphabet, length), features.one_hot(s_val, alphabet, length)),
            ("mutations", features.mutation_matrix(s_train, reference, vocab),
             features.mutation_matrix(s_val, reference, vocab)),
        ]:
            print(f"  {kind}:")
            model, _, alpha = fit_ridge(x_tr, y_tr := y_train, x_va, y_val)
            parts.append({"features": kind, "model": model, "alpha": alpha})
        bundle.update(features="ensemble", vocab=vocab, parts=parts)
        ranks = [pd.Series(p["model"].predict(x)).rank().to_numpy()
                 for p, x in zip(parts, [features.one_hot(s_val, alphabet, length),
                                         features.mutation_matrix(s_val, reference, vocab)])]
        rho = float(spearmanr(y_val, np.mean(ranks, axis=0)).statistic)
        print(f"  ensemble validation spearman={rho:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out / "bundle.joblib")
    print(f"saved {args.approach} to {args.out} (validation spearman {rho:.4f} while tuning)")


if __name__ == "__main__":
    main()

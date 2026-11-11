"""The rubric: predictions + labels + metric -> score.

This module is the only place scores come from. Project code never grades itself.
A broken submission raises SubmissionError; it is never turned into a score of zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class SubmissionError(Exception):
    """The submission could not be scored (crash, bad file, undefined metric)."""


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.ptp(y_pred) == 0:
        raise SubmissionError("all predictions are identical, so rank correlation is undefined")
    return float(stats.spearmanr(y_true, y_pred).statistic)


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.ptp(y_pred) == 0:
        raise SubmissionError("all predictions are identical, so correlation is undefined")
    return float(stats.pearsonr(y_true, y_pred).statistic)


def _ndcg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NDCG over the full ranking, linear gain.

    Relevance is the true value shifted so the minimum is 0 (targets can be negative).
    Tied predictions are ordered by input row, so avoid constant predictions.
    """
    rel = y_true - y_true.min()
    discount = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    order = np.argsort(-y_pred, kind="stable")
    ideal = float((np.sort(rel)[::-1] * discount).sum())
    if ideal == 0:
        raise SubmissionError("all true values are identical, so NDCG is undefined")
    return float((rel[order] * discount).sum() / ideal)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# name -> (function, direction)
METRICS = {
    "spearman": (_spearman, "maximize"),
    "pearson": (_pearson, "maximize"),
    "ndcg": (_ndcg, "maximize"),
    "rmse": (_rmse, "minimize"),
    "mae": (_mae, "minimize"),
}


def check_predictions(preds: pd.DataFrame, n_expected: int) -> np.ndarray:
    """Return predictions as a float array, or raise SubmissionError saying what is wrong."""
    if "prediction" not in preds.columns:
        raise SubmissionError(f"predictions file needs a 'prediction' column; found {list(preds.columns)}")
    if len(preds) != n_expected:
        raise SubmissionError(f"expected {n_expected} predictions (one per input row, same order), got {len(preds)}")
    values = pd.to_numeric(preds["prediction"], errors="coerce").to_numpy(dtype=float)
    bad = int((~np.isfinite(values)).sum())
    if bad:
        raise SubmissionError(f"{bad} predictions are missing, non-numeric or infinite")
    return values


def score(y_true, y_pred, metrics: list[str]) -> dict[str, float]:
    """Compute each named metric. Rows of y_true and y_pred must be in the same order."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; available: {sorted(METRICS)}")
    return {m: METRICS[m][0](y_true, y_pred) for m in metrics}


def is_better(new: float, old: float | None, metric: str) -> bool:
    if old is None:
        return True
    return new > old if METRICS[metric][1] == "maximize" else new < old

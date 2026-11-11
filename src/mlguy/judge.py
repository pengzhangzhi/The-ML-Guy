"""The rubric: predictions + labels + metric -> score.

This module is the only place scores come from. Project code never grades itself.
A broken submission raises SubmissionError; it is never turned into a score of zero.

A predictions file has one row per input row, in the same order, with:
  prediction   the predicted value: the number for regression, the class label for classification
  score        optional, classification only: the probability of the positive class.
               Needed by metrics that rank examples, such as roc_auc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class SubmissionError(Exception):
    """The submission could not be scored (crash, bad file, undefined metric)."""


# ---------------------------------------------------------------- regression


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


# ------------------------------------------------------------ classification


def _counts(y_true: np.ndarray, y_pred: np.ndarray, positive: float) -> tuple[int, int, int, int]:
    true_pos, pred_pos = y_true == positive, y_pred == positive
    return (int((true_pos & pred_pos).sum()), int((~true_pos & pred_pos).sum()),
            int((~true_pos & ~pred_pos).sum()), int((true_pos & ~pred_pos).sum()))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of the per-class recalls: the right metric when the classes are uneven."""
    recalls = [float(np.mean(y_pred[y_true == c] == c)) for c in np.unique(y_true)]
    return float(np.mean(recalls))


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positive = max(np.unique(y_true))
    tp, fp, _, fn = _counts(y_true, y_pred, positive)
    if tp == 0:
        return 0.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def _mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positive = max(np.unique(y_true))
    tp, fp, tn, fn = _counts(y_true, y_pred, positive)
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        raise SubmissionError("one predicted class is empty, so the correlation is undefined")
    return float((tp * tn - fp * fn) / denominator)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney form, so tied scores are handled by rank averaging."""
    positive = max(np.unique(y_true))
    is_positive = y_true == positive
    n_pos, n_neg = int(is_positive.sum()), int((~is_positive).sum())
    if n_pos == 0 or n_neg == 0:
        raise SubmissionError("only one class is present, so the area under the curve is undefined")
    ranks = stats.rankdata(y_score)
    return float((ranks[is_positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# name -> (function, direction, kind, needs the score column)
METRICS = {
    "spearman": (_spearman, "maximize", "regression", False),
    "pearson": (_pearson, "maximize", "regression", False),
    "ndcg": (_ndcg, "maximize", "regression", False),
    "rmse": (_rmse, "minimize", "regression", False),
    "mae": (_mae, "minimize", "regression", False),
    "accuracy": (_accuracy, "maximize", "classification", False),
    "balanced_accuracy": (_balanced_accuracy, "maximize", "classification", False),
    "f1": (_f1, "maximize", "classification", False),
    "mcc": (_mcc, "maximize", "classification", False),
    "roc_auc": (_roc_auc, "maximize", "classification", True),
}


def needs_score(metrics: list[str]) -> list[str]:
    return [m for m in metrics if m in METRICS and METRICS[m][3]]


def _numeric_column(preds: pd.DataFrame, name: str, n_expected: int) -> np.ndarray:
    values = pd.to_numeric(preds[name], errors="coerce").to_numpy(dtype=float)
    bad = int((~np.isfinite(values)).sum())
    if bad:
        raise SubmissionError(f"{bad} values in the '{name}' column are missing, non-numeric or infinite")
    return values


def check_predictions(preds: pd.DataFrame, n_expected: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (predictions, scores) as float arrays, or raise SubmissionError saying what is wrong."""
    if "prediction" not in preds.columns:
        raise SubmissionError(f"predictions file needs a 'prediction' column; found {list(preds.columns)}")
    if len(preds) != n_expected:
        raise SubmissionError(f"expected {n_expected} predictions (one per input row, same order), got {len(preds)}")
    values = _numeric_column(preds, "prediction", n_expected)
    scores = _numeric_column(preds, "score", n_expected) if "score" in preds.columns else None
    return values, scores


def score(y_true, y_pred, metrics: list[str], y_score=None) -> dict[str, float]:
    """Compute each named metric. Rows of y_true, y_pred and y_score must be in the same order."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; available: {sorted(METRICS)}")

    if any(METRICS[m][2] == "classification" for m in metrics):
        extra = sorted(set(y_pred.tolist()) - set(y_true.tolist()))
        if extra:
            raise SubmissionError(
                f"the 'prediction' column must hold class labels, but it contains {extra[:5]}, "
                f"which are not among the real labels {sorted(set(y_true.tolist()))}. "
                "Put probabilities in a 'score' column instead."
            )

    results = {}
    for name in metrics:
        function, _, _, wants_score = METRICS[name]
        if wants_score:
            if y_score is None:
                raise SubmissionError(f"'{name}' needs a 'score' column holding the positive-class probability")
            results[name] = function(y_true, y_score)
        else:
            results[name] = function(y_true, y_pred)
    return results


def is_better(new: float, old: float | None, metric: str) -> bool:
    if old is None:
        return True
    return new > old if METRICS[metric][1] == "maximize" else new < old

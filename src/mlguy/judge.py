"""The rubric: predictions + labels + metric -> score.

This module is the only place scores come from. Project code never grades itself.
A broken submission raises SubmissionError; it is never turned into a score of zero.

An output file has one row per example, in any order, with:
  id           which example this is, as given in the inputs
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


def check_predictions(preds: pd.DataFrame, ids: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    """Match predictions to the expected ids. Returns (predictions, scores) as float arrays in
    the order of `ids`, or raises SubmissionError saying what is wrong."""
    for column in ("id", "prediction"):
        if column not in preds.columns:
            raise SubmissionError(f"the output file needs 'id' and 'prediction' columns; found {list(preds.columns)}")
    given = preds["id"].astype(str)
    if given.duplicated().any():
        raise SubmissionError(f"{int(given.duplicated().sum())} ids appear more than once in the output file")
    expected = pd.Index([str(i) for i in ids])
    missing, extra = expected.difference(given), pd.Index(given).difference(expected)
    if len(missing) or len(extra):
        raise SubmissionError(
            f"expected one prediction for each of {len(expected)} ids; {len(missing)} are missing"
            + (f" (such as {list(missing[:3])})" if len(missing) else "")
            + (f", and {len(extra)} are not ids of this split (such as {list(extra[:3])})" if len(extra) else ""))
    preds = preds.assign(id=given).set_index("id").loc[expected]
    values = _numeric_column(preds, "prediction", len(expected))
    scores = _numeric_column(preds, "score", len(expected)) if "score" in preds.columns else None
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


def is_better(new: float, old: float | None, metric: str, direction: str | None = None) -> bool:
    """`direction` is needed only for a metric that comes from a task's own evaluator."""
    if old is None:
        return True
    return new > old if (direction or METRICS[metric][1]) == "maximize" else new < old


def check_generated(outputs: pd.DataFrame, n: int, condition_ids: list[str] | None) -> None:
    """Generated samples: n rows, or n per condition with an `id` column naming the condition
    each sample was drawn for. Other columns are free."""
    if condition_ids is None:
        if len(outputs) != n:
            raise SubmissionError(f"expected {n} generated samples, got {len(outputs)}")
        return
    if "id" not in outputs.columns:
        raise SubmissionError("generated samples need an 'id' column naming the condition each one "
                              f"was drawn for; found {list(outputs.columns)}")
    counts = outputs["id"].astype(str).value_counts()
    wrong = [c for c in condition_ids if counts.get(str(c), 0) != n]
    extra = sorted(set(counts.index) - {str(c) for c in condition_ids})
    if wrong or extra:
        raise SubmissionError(f"expected {n} samples for each of {len(condition_ids)} conditions; "
                              f"{len(wrong)} conditions have a different number"
                              + (f", and 'id' holds unknown values {extra[:5]}" if extra else ""))


# --------------------------------------------------------------------- noise
#
# A score is measured on a finite set of rows, so it would differ on another draw of them.
# Resampling the rows says by how much, and whether one submission really beats another.

RESAMPLES = 500


def _one(metric: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> float | None:
    try:
        return METRICS[metric][0](y_true, y_score if METRICS[metric][3] else y_pred)
    except SubmissionError:
        return None  # undefined on this resample (one class only, constant predictions)


def _resamples(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(RESAMPLES, n))


def interval(y_true, y_pred, metric: str, y_score=None) -> list[float] | None:
    """The 95% bootstrap interval of a score: where it would fall on another draw of the rows."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    y_score = None if y_score is None else np.asarray(y_score, dtype=float)
    values = [_one(metric, y_true[i], y_pred[i], None if y_score is None else y_score[i])
              for i in _resamples(len(y_true))]
    values = [v for v in values if v is not None and np.isfinite(v)]
    if len(values) < RESAMPLES // 2:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def chance_better(y_true, new: tuple, old: tuple, metric: str) -> float | None:
    """How often `new` beats `old` when both are scored on the same resampled rows.

    `new` and `old` are (predictions, scores-or-None). Near 0.5 means the two cannot be told
    apart on this data; near 1.0 means the improvement is real.
    """
    y_true = np.asarray(y_true, dtype=float)
    wins = total = 0.0
    for i in _resamples(len(y_true)):
        a = _one(metric, y_true[i], new[0][i], None if new[1] is None else new[1][i])
        b = _one(metric, y_true[i], old[0][i], None if old[1] is None else old[1][i])
        if a is None or b is None:
            continue
        total += 1
        wins += 1.0 if is_better(a, b, metric) else 0.5 if a == b else 0.0
    return wins / total if total >= RESAMPLES // 2 else None


def verdict(chance: float) -> str:
    """Plain words for a chance_better value."""
    chance = round(chance, 2)
    if chance >= 0.95:
        return "a real improvement"
    if chance >= 0.80:
        return "probably better, but not certain on this much data"
    if chance > 0.20:
        return "the two cannot be told apart on this data"
    if chance > 0.05:
        return "probably worse"
    return "really worse"


def mean_interval(items) -> list[float] | None:
    """The 95% interval of a score that is the mean of per-item scores."""
    items = np.asarray(items, dtype=float)
    if len(items) < 2 or not np.all(np.isfinite(items)):
        return None
    low, high = np.percentile(items[_resamples(len(items))].mean(axis=1), [2.5, 97.5])
    return [float(low), float(high)]


def mean_chance_better(new, old, direction: str) -> float | None:
    """chance_better for per-item scores, paired item by item."""
    new, old = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
    if len(new) != len(old) or len(new) < 2:
        return None
    diff = (new - old)[_resamples(len(new))].mean(axis=1)
    if direction == "minimize":
        diff = -diff
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())

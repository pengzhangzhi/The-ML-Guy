import numpy as np
import pandas as pd
import pytest

from mlguy import judge, seqio
from mlguy.judge import SubmissionError


def test_perfect_and_reversed_ranking():
    y = np.array([0.1, -0.2, 0.5, 0.3])
    assert judge.score(y, y, ["spearman", "ndcg"]) == {"spearman": pytest.approx(1.0), "ndcg": pytest.approx(1.0)}
    reversed_ = judge.score(y, -y, ["spearman", "ndcg"])
    assert reversed_["spearman"] == pytest.approx(-1.0)
    assert reversed_["ndcg"] < 1.0


def test_ndcg_matches_hand_computation():
    y_true = np.array([3.0, 1.0, 2.0])  # relevance after shift: 2, 0, 1
    y_pred = np.array([0.1, 0.9, 0.5])  # predicted order: item1, item2, item0
    d = 1 / np.log2([2, 3, 4])
    expected = (0 * d[0] + 1 * d[1] + 2 * d[2]) / (2 * d[0] + 1 * d[1] + 0 * d[2])
    assert judge.score(y_true, y_pred, ["ndcg"])["ndcg"] == pytest.approx(expected)


def test_constant_predictions_are_an_error_not_a_zero():
    with pytest.raises(SubmissionError):
        judge.score([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], ["spearman"])


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"pred": [1.0, 2.0]}),  # wrong column
        pd.DataFrame({"prediction": [1.0]}),  # wrong row count
        pd.DataFrame({"prediction": [1.0, None]}),  # missing value
        pd.DataFrame({"prediction": [1.0, "abc"]}),  # non-numeric
    ],
)
def test_bad_prediction_files_are_rejected(frame):
    with pytest.raises(SubmissionError):
        judge.check_predictions(frame, 2)


def test_is_better_respects_direction():
    assert judge.is_better(0.6, 0.5, "spearman")
    assert judge.is_better(0.4, 0.5, "rmse")
    assert judge.is_better(0.1, None, "spearman")


def test_sequence_parsing_and_checks():
    ids, seqs = seqio.parse_text(">a first\nACD\nEF\n>b\nacd\n")
    assert (ids, seqs) == (["a", "b"], ["ACDEF", "ACD"])
    assert seqio.parse_text("ACD\n\nefg\n")[1] == ["ACD", "EFG"]
    spec = {"alphabet": "ACDEFG", "length": 3}
    assert seqio.check_sequence("ACD", spec) is None
    assert "length" in seqio.check_sequence("ACDE", spec)
    assert "X" in seqio.check_sequence("ACX", spec)
    assert seqio.describe_mutations("AGD", "ACD") == "C2G"
    assert seqio.describe_mutations("ACD", "ACD") == "reference"


def test_classification_metrics_against_hand_counts():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])   # 3 positive, 5 negative
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])   # tp=2 fn=1 fp=1 tn=4
    got = judge.score(y_true, y_pred, ["accuracy", "balanced_accuracy", "f1", "mcc"])
    assert got["accuracy"] == pytest.approx(6 / 8)
    assert got["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert got["f1"] == pytest.approx(2 * (2 / 3) * (2 / 3) / (2 / 3 + 2 / 3))
    assert got["mcc"] == pytest.approx((2 * 4 - 1 * 1) / np.sqrt(3 * 3 * 5 * 5))


def test_balanced_accuracy_sees_through_a_lazy_majority_classifier():
    y_true = np.array([1] * 20 + [0] * 80)
    always_zero = np.zeros(100)
    got = judge.score(y_true, always_zero, ["accuracy", "balanced_accuracy"])
    assert got["accuracy"] == pytest.approx(0.8)
    assert got["balanced_accuracy"] == pytest.approx(0.5)


def test_roc_auc_uses_the_score_column_and_handles_ties():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert judge.score(y_true, y_pred, ["roc_auc"], y_score=np.array([0.1, 0.2, 0.3, 0.4]))["roc_auc"] == 1.0
    assert judge.score(y_true, y_pred, ["roc_auc"], y_score=np.array([0.5] * 4))["roc_auc"] == pytest.approx(0.5)
    with pytest.raises(SubmissionError, match="score"):
        judge.score(y_true, y_pred, ["roc_auc"])


def test_probabilities_in_the_prediction_column_are_rejected():
    y_true = np.array([0, 1, 0, 1])
    with pytest.raises(SubmissionError, match="class labels"):
        judge.score(y_true, np.array([0.2, 0.9, 0.4, 0.7]), ["accuracy"])


def test_check_predictions_returns_the_optional_score_column():
    values, scores = judge.check_predictions(pd.DataFrame({"prediction": [0, 1]}), 2)
    assert scores is None and list(values) == [0.0, 1.0]
    values, scores = judge.check_predictions(pd.DataFrame({"prediction": [0, 1], "score": [0.3, 0.8]}), 2)
    assert list(scores) == [0.3, 0.8]
    with pytest.raises(SubmissionError, match="score"):
        judge.check_predictions(pd.DataFrame({"prediction": [0, 1], "score": [0.3, None]}), 2)

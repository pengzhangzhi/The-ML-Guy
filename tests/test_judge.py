import numpy as np
import pandas as pd
import pytest

from mlguy import judge
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
        pd.DataFrame({"id": ["a", "b"], "pred": [1.0, 2.0]}),  # wrong column
        pd.DataFrame({"id": ["a"], "prediction": [1.0]}),  # an example is missing
        pd.DataFrame({"prediction": [1.0, 2.0]}),  # no ids
        pd.DataFrame({"id": ["a", "b"], "prediction": [1.0, None]}),  # missing value
        pd.DataFrame({"id": ["a", "b"], "prediction": [1.0, "abc"]}),  # non-numeric
    ],
)
def test_bad_prediction_files_are_rejected(frame):
    with pytest.raises(SubmissionError):
        judge.check_predictions(frame, ["a", "b"])


def test_is_better_respects_direction():
    assert judge.is_better(0.6, 0.5, "spearman")
    assert judge.is_better(0.4, 0.5, "rmse")
    assert judge.is_better(0.1, None, "spearman")


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


def test_check_predictions_matches_by_id_and_returns_the_optional_score():
    values, scores = judge.check_predictions(pd.DataFrame({"id": ["b", "a"], "prediction": [1, 0]}), ["a", "b"])
    assert scores is None and list(values) == [0.0, 1.0], "returned in the order of the expected ids"
    values, scores = judge.check_predictions(
        pd.DataFrame({"id": [2, 1], "prediction": [1, 0], "score": [0.8, 0.3]}), ["1", "2"])
    assert list(scores) == [0.3, 0.8], "ids are compared as text, so 1 and '1' are the same example"
    with pytest.raises(SubmissionError, match="score"):
        judge.check_predictions(pd.DataFrame({"id": ["a", "b"], "prediction": [0, 1], "score": [0.3, None]}), ["a", "b"])
    with pytest.raises(SubmissionError, match="1 are missing"):
        judge.check_predictions(pd.DataFrame({"id": ["a", "x"], "prediction": [0, 1]}), ["a", "b"])
    with pytest.raises(SubmissionError, match="more than once"):
        judge.check_predictions(pd.DataFrame({"id": ["a", "a"], "prediction": [0, 1]}), ["a", "b"])


def test_multiclass_text_labels():
    y = np.array(["alpha", "beta", "beta", "delta", "delta", "delta"])
    pred = np.array(["alpha", "beta", "delta", "delta", "delta", "gamma"])   # gamma never occurs: just wrong
    got = judge.score(y, pred, ["accuracy", "balanced_accuracy", "f1", "mcc"])
    assert got["accuracy"] == pytest.approx(4 / 6)
    assert got["balanced_accuracy"] == pytest.approx((1 + 0.5 + 2 / 3) / 3)
    assert 0 < got["f1"] < 1 and 0 < got["mcc"] < 1
    with pytest.raises(SubmissionError, match="two classes"):
        judge.score(y, pred, ["roc_auc"], y_score=np.zeros(6))
    values, _ = judge.check_predictions(pd.DataFrame({"id": ["a", "b"], "prediction": ["beta", "alpha"]}),
                                        ["a", "b"], numeric=False)
    assert list(values) == ["beta", "alpha"]


def test_binary_metrics_unchanged_and_labels_match_across_types():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    pred = np.array(["1", "1", "0", "1", "0", "0", "0", "0"])          # text that holds numbers
    got = judge.score(y, pred, ["accuracy", "f1", "mcc"])
    assert got["accuracy"] == pytest.approx(6 / 8)
    assert got["f1"] == pytest.approx(2 / 3)
    assert got["mcc"] == pytest.approx((2 * 4 - 1 * 1) / np.sqrt(3 * 3 * 5 * 5))

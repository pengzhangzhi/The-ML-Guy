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

from __future__ import annotations

import pandas as pd
import pytest

from mlguy import judge
from mlguy.judge import SubmissionError


def test_is_better_respects_direction():
    assert judge.is_better(0.6, 0.5, "maximize")
    assert judge.is_better(0.4, 0.5, "minimize")
    assert judge.is_better(0.1, None, "maximize")


def test_generated_samples_are_counted_per_condition():
    judge.check_generated(pd.DataFrame({"id": ["a", "a", "b", "b"]}), 2, ["a", "b"])
    with pytest.raises(SubmissionError, match="different number"):
        judge.check_generated(pd.DataFrame({"id": ["a", "b", "b"]}), 2, ["a", "b"])

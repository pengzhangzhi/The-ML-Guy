"""Checks on the split, run once by `mlguy init` before the task is frozen.

These are the mistakes that make a score mean less than it appears to, and that are cheap to
detect mechanically. They are warnings, not refusals: a published split may be reproduced on
purpose. Each one should be fixed, or explained in the brief.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

SMALL_N = 100
BALANCE_GAP = 0.05
SHIFT_KS = 0.2
BASE_RATE_METRICS = ("accuracy", "f1")


def _overlap(a: pd.DataFrame, b: pd.DataFrame, columns: list[str]) -> int:
    left = a[columns].astype(str).drop_duplicates()
    return int(len(left.merge(b[columns].astype(str).drop_duplicates(), on=columns)))


def split_checks(task: dict, train: pd.DataFrame, validation: pd.DataFrame | None,
                 test: pd.DataFrame | None, compare_targets: bool = True) -> list[str]:
    """`validation` and `test` may be absent (a generative task need not have them), and targets
    are compared only when a built-in metric scores against a label column."""
    columns, target = list(task["data"].get("input_columns", [])), task["data"].get("target_column")
    declared_mismatch = not task["data"].get("validation_matches_test", True)
    found = []
    if task["task"]["type"] == "generation":
        return found  # conditions legitimately repeat across splits, and the sample size is set by the task
    parts = [(n, f) for n, f in (("train", train), ("validation", validation), ("test", test)) if f is not None]

    for i, (name_a, a) in enumerate(parts):
        for name_b, b in parts[i + 1:]:
            shared = _overlap(a, b, columns)
            if shared:
                found.append(f"{shared} input rows appear in both {name_a} and {name_b}, so the "
                             f"{name_b} score partly measures memorisation.")

    for name, part in parts[1:]:
        if len(part) < SMALL_N:
            found.append(f"{name} has only {len(part)} rows, so its score will be noisy; "
                         "small differences between submissions will not mean much.")

    if declared_mismatch or not compare_targets or validation is None or test is None:
        return found  # the difference between validation and test is already on record

    if task["task"]["type"] == "classification":
        classes = sorted(set(validation[target]) | set(test[target]))
        gaps = {c: abs(float((validation[target] == c).mean()) - float((test[target] == c).mean())) for c in classes}
        worst = max(gaps, key=gaps.get)
        if gaps[worst] > BALANCE_GAP:
            message = (f"class {worst} is {(validation[target] == worst).mean():.0%} of validation but "
                       f"{(test[target] == worst).mean():.0%} of test.")
            if task["metric"]["primary"] in BASE_RATE_METRICS:
                message += (f" '{task['metric']['primary']}' moves with the class balance, so the search "
                            "would be ranked by the wrong thing; prefer balanced_accuracy or mcc.")
            found.append(message + " Set validation_matches_test = false if this is intended.")
    else:
        shift = float(stats.ks_2samp(validation[target], test[target]).statistic)
        if shift > SHIFT_KS:
            found.append(f"the target is distributed differently in validation and test (KS distance "
                         f"{shift:.2f}), which suggests they ask different questions. Redraw validation "
                         "to resemble test, or set validation_matches_test = false and say why.")
    return found

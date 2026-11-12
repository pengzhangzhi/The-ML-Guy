"""Checks on the split, run once by `mlguy init` before the task is frozen.

These are the mistakes that make a score mean less than it appears to, and that are cheap to
detect mechanically. They are warnings, not refusals: a published split may be reproduced on
purpose. Each one should be fixed, or explained in the brief.

The framework does not read a task's inputs, so what it can compare is ids and labels. The
one exception: when the inputs happen to be CSV tables, it also looks for the same row
appearing in two splits, because that is the most common way a score ends up flattering.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

SMALL_N = 100
BALANCE_GAP = 0.05
SHIFT_KS = 0.2


def _shared_rows(a: Path, b: Path, ignore: set[str]) -> int | None:
    """Rows with identical content in two CSV tables, compared on the columns they share."""
    if not (a.suffix == b.suffix == ".csv" and a.is_file() and b.is_file()):
        return None
    left, right = pd.read_csv(a, dtype=str), pd.read_csv(b, dtype=str)
    columns = [c for c in left.columns if c in right.columns and c not in ignore]
    if not columns:
        return None
    return int(len(left[columns].drop_duplicates().merge(right[columns].drop_duplicates(), on=columns)))


def split_checks(task: dict, validation: pd.DataFrame | None, test: pd.DataFrame | None,
                 inputs: dict[str, Path], id_column: str, target_column: str,
                 compare_targets: bool = True) -> list[str]:
    """`validation` and `test` are the label tables (id, and target when they carry one);
    `inputs` maps train/validation/test to wherever each split's inputs are stored."""
    found = []
    generation = task["task"]["type"] == "generation"
    noun = "conditions" if generation else "examples"

    names = [n for n in ("train", "validation", "test") if inputs.get(n) is not None]
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            shared = _shared_rows(inputs[name_a], inputs[name_b], {id_column, target_column})
            if shared:
                found.append(f"{shared} rows have identical inputs in {name_a} and {name_b}, so the "
                             f"{name_b} score partly measures memorisation.")
    if validation is not None and test is not None:
        both = len(set(validation[id_column].astype(str)) & set(test[id_column].astype(str)))
        if both:
            found.append(f"{both} ids appear in both validation and test.")

    for name, part in [("validation", validation), ("test", test)]:
        if part is not None and len(part) < SMALL_N:
            found.append(f"{name} has only {len(part)} {noun}, so its score will be noisy; "
                         "small differences between submissions will not mean much.")

    declared_mismatch = not task["data"].get("validation_matches_test", True)
    # A generation task's conditions carry no label to compare, so the distribution checks below
    # do not apply to them. The ones above do: a condition in two splits is still leakage, and
    # too few of them is still a noisy score, whatever is sampled for each one.
    if generation or declared_mismatch or not compare_targets or validation is None or test is None:
        return found  # the difference between validation and test is already on record

    target = target_column
    if task["task"]["type"] == "classification":
        classes = sorted(set(validation[target]) | set(test[target]))
        gaps = {c: abs(float((validation[target] == c).mean()) - float((test[target] == c).mean())) for c in classes}
        worst = max(gaps, key=gaps.get)
        if gaps[worst] > BALANCE_GAP:
            message = (f"class {worst} is {(validation[target] == worst).mean():.0%} of validation but "
                       f"{(test[target] == worst).mean():.0%} of test.")
            message += (" A metric that moves with the class balance, such as plain accuracy, would rank the "
                        "search by the wrong thing.")
            found.append(message + " Set validation_matches_test = false if this is intended.")
    else:
        shift = float(stats.ks_2samp(validation[target], test[target]).statistic)
        if shift > SHIFT_KS:
            found.append(f"the target is distributed differently in validation and test (KS distance "
                         f"{shift:.2f}), which suggests they ask different questions. Redraw validation "
                         "to resemble test, or set validation_matches_test = false and say why.")
    return found

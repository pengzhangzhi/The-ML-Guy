"""What the framework itself checks about outputs. The score comes from the task's evaluator.

A broken submission raises SubmissionError; it is never turned into a score of zero.
"""

from __future__ import annotations

import pandas as pd


class SubmissionError(Exception):
    """The submission could not be scored (crash, bad file, no usable score)."""


def is_better(new: float, old: float | None, direction: str) -> bool:
    if old is None:
        return True
    return new > old if direction == "maximize" else new < old


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

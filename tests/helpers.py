"""Shared by the tests: write full tables out in the layout a task uses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_TOML = '''\
train = "data/train.csv"
validation_inputs = "data/validation_inputs.csv"
validation_labels = "data/validation_labels.csv"
test_inputs = "data/test_inputs.csv"
test_labels = "data/test_labels.csv"
'''


def write_splits(data_dir: Path, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
                 target: str = "target") -> None:
    """Give every example an id, and split validation and test into inputs and labels."""
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / "train.csv", index=False)
    for name, part in (("validation", validation), ("test", test)):
        part = part.reset_index(drop=True)
        part.insert(0, "id", [f"{name[0]}{i}" for i in range(len(part))])
        part.drop(columns=[target]).to_csv(data_dir / f"{name}_inputs.csv", index=False)
        part[["id", target]].to_csv(data_dir / f"{name}_labels.csv", index=False)

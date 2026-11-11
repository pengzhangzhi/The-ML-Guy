"""Fetch the DeepSol solubility split and write this project's task data.

The published split is used as it stands. `mlguy init` then hides the test labels.

    python make_task.py && mlguy init
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
BASE = "https://huggingface.co/datasets/proteinea/solubility/resolve/main"
PARTS = {"train": "solubility_training.csv", "validation": "solubility_validation.csv",
         "test": "solubility_testing.csv"}


def main() -> None:
    raw = HERE / "raw"
    raw.mkdir(exist_ok=True)
    data_dir = HERE / "task" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, filename in PARTS.items():
        path = raw / filename
        if not path.exists():
            print(f"downloading {filename}")
            urllib.request.urlretrieve(f"{BASE}/{filename}", path)
        df = pd.read_csv(path).rename(columns={"sequences": "sequence", "labels": "target"})
        assert set(df["target"]) <= {0, 1}, "labels must be 0 or 1"
        assert df["sequence"].str.len().min() > 0, "empty sequence"
        df[["sequence", "target"]].to_csv(data_dir / f"{name}.csv", index=False)
        share = df["target"].mean()
        print(f"{name:<11} {len(df):>6} sequences, {share:.0%} soluble, "
              f"lengths {df['sequence'].str.len().min()}-{df['sequence'].str.len().max()}")

    print("next: mlguy init")


if __name__ == "__main__":
    main()

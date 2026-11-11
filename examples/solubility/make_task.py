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


def write_split(data_dir: Path, name: str, part: pd.DataFrame) -> None:
    """Training keeps inputs and labels together; validation and test are split into the two."""
    if name == "train":
        part[["sequence", "target"]].to_csv(data_dir / "train.csv", index=False)
        return
    part = part.reset_index(drop=True)
    part.insert(0, "id", [f"{name}-{i:05d}" for i in range(len(part))])
    part[["id", "sequence"]].to_csv(data_dir / f"{name}_inputs.csv", index=False)
    part[["id", "target"]].to_csv(data_dir / f"{name}_labels.csv", index=False)


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
        write_split(data_dir, name, df)
        share = df["target"].mean()
        print(f"{name:<11} {len(df):>6} sequences, {share:.0%} soluble, "
              f"lengths {df['sequence'].str.len().min()}-{df['sequence'].str.len().max()}")

    print("next: mlguy init")


if __name__ == "__main__":
    main()

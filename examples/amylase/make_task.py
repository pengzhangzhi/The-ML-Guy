"""Turn the published FLIP2 split file into this project's task data.

Writes task/data: train.csv, and inputs and labels for validation and test. `mlguy init` then
moves the test labels out of the project folder.

    python make_task.py && mlguy init
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from mlguy.project import load_task

HERE = Path(__file__).parent


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
    task = load_task(HERE)
    seq, tgt = "sequence", "target"

    raw = HERE / "raw" / "one_to_many.csv.gz"
    if not raw.exists():
        raw.parent.mkdir(exist_ok=True)
        print(f"downloading {task['data']['source']}")
        urllib.request.urlretrieve(task["data"]["source"], raw)

    df = pd.read_csv(raw)
    is_val = df["validation"].astype(str).str.lower() == "true"
    train = df[(df["set"] == "train") & ~is_val]
    val = df[(df["set"] == "train") & is_val]
    test = df[df["set"] == "test"]
    assert len(train) + len(val) + len(test) == len(df), "rows outside train/validation/test"
    assert df[seq].is_unique, "duplicate sequences"
    assert task["input"]["reference"] in set(train[seq]), "reference sequence not found in training data"

    data_dir = HERE / "task" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        write_split(data_dir, name, part)

    print(f"train {len(train)}  validation {len(val)}  test {len(test)}")
    print("next: mlguy init   (it hides the test labels outside the project)")


if __name__ == "__main__":
    main()

"""Turn the published FLIP2 split file into this project's task data.

Writes task/data/{train,validation,test}.csv. `mlguy init` then hides the test labels
outside the project folder. In later milestones the builder skill does this step.

    python make_task.py && mlguy init
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from mlguy.project import load_task

HERE = Path(__file__).parent


def main() -> None:
    task = load_task(HERE)
    seq, tgt = task["data"]["input_columns"][0], task["data"]["target_column"]

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
        part[[seq, tgt]].to_csv(data_dir / f"{name}.csv", index=False)

    print(f"train {len(train)}  validation {len(val)}  test {len(test)}")
    print("next: mlguy init   (it hides the test labels outside the project)")


if __name__ == "__main__":
    main()

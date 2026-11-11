"""The project's one fixed interface: run the model on some sequences.

    python infer.py --model model --input sequences.csv --output outputs.csv

--input is a CSV with `id` and `sequence` columns. --output is a CSV with one row per id:
`prediction`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import features  # noqa: E402



def predict(bundle: dict, seqs: list[str]) -> np.ndarray:
    if bundle["features"] == "ensemble":
        ranks = []
        for part in bundle["parts"]:
            x = features.build(seqs, {**bundle, "features": part["features"]})
            ranks.append(pd.Series(part["model"].predict(x)).rank().to_numpy())
        return np.mean(ranks, axis=0)
    return bundle["model"].predict(features.build(seqs, bundle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model / "bundle.joblib")
    rows = pd.read_csv(args.input)
    seqs = rows["sequence"].astype(str).tolist()
    pd.DataFrame({"id": rows["id"], "prediction": predict(bundle, seqs)}).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

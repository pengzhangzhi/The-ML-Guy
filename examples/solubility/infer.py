"""The project's one fixed interface: run the model on some sequences.

    python infer.py --model model --input sequences.csv --output outputs.csv

--input is a CSV with `id` and `sequence` columns. --output is a CSV with one row per id:
`prediction` (the class label) and `score` (the probability of being soluble).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import features  # noqa: E402



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model / "bundle.joblib")
    rows = pd.read_csv(args.input)
    seqs = rows["sequence"].astype(str).tolist()
    x = features.build(seqs, bundle)
    if bundle.get("scaler") is not None:
        x = bundle["scaler"].transform(x)

    model = bundle["model"]
    out = {"id": rows["id"], "prediction": model.predict(x)}
    if bundle.get("classification"):
        positive = max(model.classes_)
        out["score"] = model.predict_proba(x)[:, list(model.classes_).index(positive)]
    pd.DataFrame(out).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

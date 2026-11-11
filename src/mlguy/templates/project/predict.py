"""The project's one fixed interface: a table in, a prediction per row out.

    python predict.py --model model --input rows.csv --output predictions.csv

The input CSV holds this task's input columns. The output CSV has one row per input row, in
the same order, with:
  prediction   the predicted value, or the class label if this is a classification task
  score        classification only: the probability of the positive class

Everything inside this file is yours to change. The command line is not: the rubric and the
project's page both call it exactly as above.
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
    x = features.build(rows, bundle["spec"])
    if bundle.get("scaler") is not None:
        x = bundle["scaler"].transform(x)

    model = bundle["model"]
    out = {"prediction": model.predict(x)}
    if bundle.get("classification"):
        positive = max(model.classes_)
        out["score"] = model.predict_proba(x)[:, list(model.classes_).index(positive)]
    pd.DataFrame(out).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

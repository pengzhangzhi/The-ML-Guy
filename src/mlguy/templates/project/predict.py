"""The project's one fixed interface: sequences in, predictions out.

    python predict.py --model model --input sequences.csv --output predictions.csv

Input is a CSV with a sequence column, or a FASTA file. Output is a CSV with one
'prediction' per input row, in the same order.

Everything inside this file is yours to change. The command line is not: the rubric and the
prediction page both call it exactly as above.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import features  # noqa: E402

from mlguy.seqio import read_sequences  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model / "bundle.joblib")
    _, seqs = read_sequences(args.input)
    predictions = bundle["model"].predict(features.build(seqs, bundle))
    pd.DataFrame({"sequence": seqs, "prediction": predictions}).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

"""The project's one fixed interface: run the model on some inputs.

    python infer.py --model model --input PATH --output outputs.csv

--input is in whatever form this task stores its inputs (see task/task.toml): a table, a
matrix, a folder of files. --output is a CSV with one row per example:

    id           which example this is, as given in the inputs
    prediction   the predicted number, or the class label for classification
    score        classification only, optional: the probability of the positive class

The command line and the output columns are fixed, because `mlguy` and the project's page both
call this file. Everything else about it, and everything in src/, is yours.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit("infer.py is still the empty template: load the model from --model, read --input, "
                     "and write id and prediction to --output")


if __name__ == "__main__":
    main()

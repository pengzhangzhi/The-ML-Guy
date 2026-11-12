"""The project's one fixed interface: run the model on some inputs.

    python infer.py --model model --input PATH --output outputs.csv

--input is in whatever form this task stores its inputs (see task/task.toml): a table, a
matrix, a folder of files. --output is a CSV with an `id` column naming the example each row
is for, and whatever else task/evaluate.py reads (by convention `prediction`).

A generation task is called with two more arguments, which it must accept:

    python infer.py --model model --input CONDITIONS --n 8 --seed 0 --output outputs.csv

--n is how many samples to draw for each condition in --input, and --seed makes that draw
reproducible; both come from [sampling] in task.toml, and the final check uses different ones.
Write n rows per condition, each carrying that condition's id. The framework refuses outputs
that hold a different number, so a submission cannot quietly be the best few of many.

Whichever kind it is, this file must not read anything out of task/. It is scored from a copy
holding only infer.py, src/ and model/, and the user runs it from final/, which has no task/
either: anything it needs at prediction time belongs in model/.

The command line is fixed, because `mlguy` and the project's page both call this file.
Everything else about it, and everything in src/, is yours.
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

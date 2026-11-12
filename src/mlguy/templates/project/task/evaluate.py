"""The task's scoring: the model's outputs and the labels in, numbers out. Frozen with the task.

    python evaluate.py --outputs FILE --reference LABELS --inputs PATH \
                       --split validation|final --result FILE

--outputs is what infer.py wrote; --reference is that split's labels CSV (id and target);
--inputs is the same inputs the model was given, for a score that depends on the example itself.
Write {"metrics": {"name": number, ...}} to --result, including task.toml's [metric] primary.
Refuse outputs that cannot be scored (missing or repeated ids, wrong columns) by exiting with
an error, rather than scoring them as zero.

Write nothing except --result, and open every path you are given as it was given. This script
runs with task/ as its working directory, so a relative path it writes to lands in the frozen
task/ and changes its fingerprint, which stops the next submission from scoring; and --result is
the only file anything reads back, so a diagnostic written beside it is thrown away. Put
diagnostics in the metrics, or take an explicit path for them.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--inputs")
    parser.add_argument("--split", choices=["validation", "final"], required=True)
    parser.add_argument("--result", required=True)
    parser.parse_args()
    raise SystemExit("evaluate.py is still the empty template: score --outputs against --reference "
                     "and write the metrics to --result")


if __name__ == "__main__":
    main()

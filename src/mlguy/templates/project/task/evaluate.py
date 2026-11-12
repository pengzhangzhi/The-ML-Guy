"""The task's scoring: the model's outputs and the labels in, numbers out. Frozen with the task.

    python evaluate.py --outputs FILE --reference LABELS --split validation|final --result FILE

--outputs is what infer.py wrote; --reference is that split's labels CSV (id and target).
Write {"metrics": {"name": number, ...}} to --result, including task.toml's [metric] primary.
Refuse outputs that cannot be scored (missing or repeated ids, wrong columns) by exiting with
an error, rather than scoring them as zero.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--split", choices=["validation", "final"], required=True)
    parser.add_argument("--result", required=True)
    parser.parse_args()
    raise SystemExit("evaluate.py is still the empty template: score --outputs against --reference "
                     "and write the metrics to --result")


if __name__ == "__main__":
    main()

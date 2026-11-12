"""The evaluator contract, with one function left for you. Copy it to task/evaluate.py.

    python evaluate.py --outputs PATH [--reference PATH] [--inputs PATH] \
                       --split validation|final --result FILE

--outputs is whatever infer.py wrote, a file or a folder. --reference and --inputs are the
split's, passed when task.toml declares them. Everything here except `score` is the contract and
is the same for every task; `score` is the only place that knows what this problem means.
"""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path


class Refused(Exception):
    """These outputs cannot be scored. They become an error, never a score of zero: a zero is a
    result the search will try to beat, and a model that crashed is not a model that scored badly."""


def score(outputs: Path, reference: Path | None, inputs: Path | None, split: str) -> dict[str, float]:
    """>>> REPLACE THIS. Everything else is the contract. <<<

    Return every number worth reporting, with task.toml's `[metric] primary` among them. Include
    the ones that can disagree with the primary: a metric left out is one the user never sees.
    Raise Refused for anything that cannot be scored: missing or extra items, the wrong form,
    the wrong count of anything task.toml fixes. Be deterministic, and do not depend on order.
    """
    raise Refused("evaluate.py is still the template: write score()")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--split", choices=["validation", "final"], required=True)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()

    try:
        metrics = score(args.outputs, args.reference, args.inputs, args.split)
    except Refused as e:
        raise SystemExit(f"refused: {e}")

    # Catch a mistake here rather than as a confusing error from mlguy: the primary must be
    # reported and must be a number. task.toml sits beside this file.
    primary = tomllib.loads((Path(__file__).parent / "task.toml").read_text())["metric"]["primary"]
    if primary not in metrics or not math.isfinite(metrics[primary]):
        raise SystemExit(f"score() must report a finite {primary!r}; it reported {metrics}")

    # --result is the only file anything reads back, and this script runs with task/ as its
    # working directory, so a relative path written anywhere else lands in the frozen task and
    # breaks its fingerprint. Write this one file and nothing else.
    args.result.write_text(json.dumps({"metrics": {k: float(v) for k, v in metrics.items()}}))


if __name__ == "__main__":
    main()

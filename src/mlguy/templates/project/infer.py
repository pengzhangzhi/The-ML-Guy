"""The project's one fixed interface: run the model.

    python infer.py --model model [--input PATH] --output PATH [the task's [infer] args]

--input is a split's inputs exactly as task/task.toml stores them: a table, a folder, anything.
A task with no inputs omits it. --output is where to write what the model returns, as a file or
a folder, in whatever form task/evaluate.py reads. Any [infer] args in task.toml are passed as
--key value (say --n 20 --seed 0) and must be accepted.

This file must not read anything out of task/. It is scored from a copy holding only infer.py,
src/ and model/, and the user runs it from final/, which has no task/ either: anything it needs
at run time belongs in model/.

The command line is fixed, because `mlguy` and the project's page both call this file.
Everything else about it, and everything in src/, is yours.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args, extra = parser.parse_known_args()
    raise SystemExit("infer.py is still the empty template: load the model from --model, read "
                     "--input, and write what the model returns to --output")


if __name__ == "__main__":
    main()

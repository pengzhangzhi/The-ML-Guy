"""The worked examples are documentation that runs, so they must at least still import.

They are not exercised end to end here: that needs their data. What this catches is an
example left behind by a change to the framework.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(p for p in EXAMPLES.glob("*/*.py")) + sorted(p for p in EXAMPLES.glob("*/src/*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_example_script_imports(script, monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location(f"example_{script.parent.name}_{script.stem}", script)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        if any(name in str(e) for name in ("gradio", "torch", "datasets")):
            pytest.skip(f"optional library for this example: {e}")
        raise
    finally:
        sys.modules.pop(spec.name, None)


def test_every_example_carries_its_own_evaluator():
    for project in sorted(p for p in EXAMPLES.iterdir() if (p / "task" / "task.toml").exists()):
        evaluator = project / "task" / "evaluate.py"
        assert evaluator.exists(), f"{project.name} has no task/evaluate.py"
        assert "--inputs" in evaluator.read_text(), f"{project.name}'s evaluator must accept --inputs"

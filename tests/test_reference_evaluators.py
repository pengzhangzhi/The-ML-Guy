"""The worked evaluators beside the build skill are the contract made concrete.

An agent copies them, so they have to stay valid as the contract moves. These checks are cheap
and catch the thing that actually goes wrong: a reference left behind by a change to the CLI.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REFERENCE = Path(__file__).resolve().parents[1] / "skills" / "build-project" / "reference"
SCRIPTS = sorted(REFERENCE.glob("evaluate_*.py"))
ARGUMENTS = ("--outputs", "--reference", "--inputs", "--split", "--result")


def test_there_is_a_worked_evaluator_for_both_kinds_of_task():
    assert {p.name for p in SCRIPTS} == {"evaluate_prediction.py", "evaluate_generation.py"}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_a_reference_evaluator_accepts_every_argument_the_framework_passes(script):
    source = script.read_text()
    ast.parse(source)
    for argument in ARGUMENTS:
        assert f'"{argument}"' in source, f"{script.name} must accept {argument}, even if it ignores it"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_a_reference_evaluator_refuses_rather_than_scoring_zero(script):
    source = script.read_text()
    assert "SystemExit" in source, f"{script.name} must refuse unscoreable outputs with an error"
    assert "raise NotImplementedError" in source or ">>> REPLACE THIS" in source, \
        f"{script.name} must mark the one part the agent replaces"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_a_reference_evaluator_runs_far_enough_to_report_its_arguments(script):
    done = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    for argument in ARGUMENTS:
        assert argument in done.stdout


def test_the_reference_brief_ends_with_the_line_init_looks_for():
    """The brief template lives with the skill, not in the framework, so nothing but this test
    ties its last line to the one `mlguy init` refuses without."""
    from mlguy.cli import APPROVAL
    brief = (REFERENCE / "brief.md").read_text()
    assert any(line.startswith(APPROVAL) for line in brief.splitlines())

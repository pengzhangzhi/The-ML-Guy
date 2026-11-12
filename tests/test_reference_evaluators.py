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

SKILLS = Path(__file__).resolve().parents[1] / "skills"
REFERENCE = SKILLS / "build-project" / "reference"
EVALUATION = SKILLS / "evaluation"
SCRIPTS = [EVALUATION / "reference" / "evaluate.py", *sorted((EVALUATION / "cases").glob("evaluate_*.py"))]
ARGUMENTS = ("--outputs", "--reference", "--inputs", "--split", "--result")


def test_the_evaluation_skill_has_a_template_and_worked_cases():
    """The template is the contract for any task; each case is one shape of problem, and every
    case evaluator a case note points to exists."""
    assert all(p.exists() for p in SCRIPTS) and len(SCRIPTS) >= 2
    for note in (EVALUATION / "cases").glob("*.md"):
        for name in (w.strip("`.,") for w in note.read_text().split() if w.strip("`.,").endswith(".py")):
            assert (EVALUATION / "cases" / name).exists(), f"{note.name} points to a missing {name}"


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


def test_every_skill_that_writes_for_the_user_points_at_the_writing_skill():
    """A skill nothing points to does not get loaded, so the pointers are what make it apply."""
    skills = SKILLS
    assert (skills / "writing" / "SKILL.md").exists()
    for name in ("build-project", "audit", "research", "deploy"):
        assert "`writing` skill" in " ".join((skills / name / "SKILL.md").read_text().split()), name


def test_the_build_skill_hands_the_machine_to_the_environment_skill():
    skills = REFERENCE.parents[1]
    assert (skills / "environment" / "SKILL.md").exists()
    assert "`environment` skill" in " ".join((skills / "build-project" / "SKILL.md").read_text().split())


def test_the_build_skill_hands_success_to_the_evaluation_skill():
    text = " ".join((SKILLS / "build-project" / "SKILL.md").read_text().split())
    assert "`evaluation` skill" in text
    assert "## 5. Write the evaluator" not in text, "the method lives in one place"


def test_the_main_skills_hold_no_task_kinds():
    """Lessons about particular shapes of problem live in evaluation/cases/, opened only when a
    problem looks like them. The skills every project reads stay general."""
    for name in ("build-project", "research", "audit", "evaluation"):
        text = (SKILLS / name / "SKILL.md").read_text()
        for word in ("[sampling]", "test_labels", "validation_labels", "generative task", "per condition"):
            assert word not in text, f"{name} still says {word!r}"

"""The loop end to end on a tiny made-up project: init, validate, submit, budget, finish."""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from helpers import DATA_TOML, write_splits
from mlguy.cli import main
from mlguy.records import Experiments

TASK_TOML = """\
schema_version = "0.1"
name = "tiny-test-project"

[task]
type = "regression"
input = "protein_sequence"
output = "activity"
higher_is_better = true

[input]
alphabet = "ACDE"
length = 8

[data]
""" + DATA_TOML + """
[metric]
primary = "spearman"
secondary = ["ndcg"]

[budget]
max_submissions = 2
"""

PREDICT_PY = '''\
import argparse, json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, required=True)
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()

scale = json.loads((a.model / "weights.json").read_text())["scale"]
rows = pd.read_csv(a.input)
rows["prediction"] = [scale * s.count("A") + {offset} for s in rows["sequence"]]
rows[["id", "prediction"]].to_csv(a.output, index=False)
'''


def _write_model(project: Path, scale: float) -> None:
    (project / "model").mkdir(exist_ok=True)
    (project / "model" / "weights.json").write_text(json.dumps({"scale": scale}))


def _write_predict(project: Path, offset: str = "0") -> None:
    (project / "infer.py").write_text(PREDICT_PY.format(offset=offset))


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "state"))
    project = tmp_path / "project"
    (project / "task" / "data").mkdir(parents=True)
    (project / "task" / "task.toml").write_text(TASK_TOML)

    rng = random.Random(0)
    rows = []
    for _ in range(120):
        seq = "".join(rng.choice("ACDE") for _ in range(8))
        rows.append({"sequence": seq, "target": seq.count("A") + rng.random()})
    frame = pd.DataFrame(rows).drop_duplicates("sequence")
    write_splits(project / "task" / "data", frame[:60], frame[60:90], frame[90:])

    _write_predict(project)
    _write_model(project, 1.0)
    return project


def run(project: Path, *args: str) -> int:
    return main([args[0], "--project", str(project), *args[1:]])


def test_full_loop(project, capsys):
    assert run(project, "init") == 0
    assert not (project / "task" / "data" / "test_labels.csv").exists(), "the test labels should be gone"
    assert (project / "task" / "data" / "test_inputs.csv").exists()
    assert "target" not in pd.read_csv(project / "task" / "data" / "test_inputs.csv").columns

    assert run(project, "validate") == 0
    experiments = Experiments(project)
    assert [r["number"] for r in experiments.all()] == [0]
    assert experiments.all()[0]["counts_against_budget"] is False

    # a submission that scores worse
    _write_model(project, -1.0)
    assert run(project, "submit", "--note", "flip the sign") == 0
    assert experiments.best("spearman")["number"] == 0

    # a submission that scores the same as the baseline but is still recorded
    _write_model(project, 2.0)
    assert run(project, "submit", "--note", "double the scale") == 0
    assert len(experiments.all()) == 3

    # budget of 2 submissions is now spent
    assert run(project, "submit", "--note", "one too many") == 2
    assert "refused" in capsys.readouterr().err

    assert run(project, "finish") == 0
    report = (project / "REPORT.md").read_text()
    assert "flip the sign" in report and "test" in report.lower()
    kept = json.loads((project / "final" / "model" / "weights.json").read_text())
    assert kept == {"scale": 1.0}, "final/ holds the best submission, not whatever was tried last"
    assert run(project, "finish") == 2, "the test set must not be scored twice"


def test_broken_submission_is_an_error_not_a_zero(project, capsys):
    run(project, "init")
    run(project, "validate")
    (project / "infer.py").write_text("raise SystemExit('boom')")
    assert run(project, "submit", "--note", "break it") == 1
    record = Experiments(project).all()[-1]
    assert record["status"] == "error" and "metrics" not in record
    assert record["counts_against_budget"] is True
    assert Experiments(project).best("spearman")["number"] == 0


def test_changing_the_task_is_refused(project, capsys):
    run(project, "init")
    run(project, "validate")
    toml = project / "task" / "task.toml"
    toml.write_text(toml.read_text().replace('primary = "spearman"', 'primary = "pearson"'))
    assert run(project, "submit", "--note", "quietly change the metric") == 2
    assert "task/ has changed" in capsys.readouterr().err


def test_validate_refuses_before_a_model_exists(project, capsys):
    run(project, "init")
    (project / "model" / "weights.json").unlink()
    (project / "model").rmdir()
    assert run(project, "validate") == 1
    assert "model" in capsys.readouterr().err


def test_submit_requires_validate_first(project, capsys):
    run(project, "init")
    assert run(project, "submit", "--note", "skip the gate") == 2
    assert "validate" in capsys.readouterr().err


def test_status_json_reports_best_and_budget(project, capsys):
    run(project, "init")
    run(project, "validate")
    capsys.readouterr()
    assert run(project, "status", "--json") == 0
    status = json.loads(capsys.readouterr().out)
    assert status["best"]["number"] == 0
    assert status["budget"]["limits"]["submissions"] == {"used": 0, "limit": 2}
    assert status["test_scored"] is None


def test_dashboard_renders_without_error(project, monkeypatch):
    """Run the dashboard script against a stub streamlit, to catch mistakes in the page itself."""
    from unittest.mock import MagicMock

    run(project, "init")
    run(project, "validate")
    _write_model(project, 3.0)
    run(project, "submit", "--note", "triple the scale")

    stub = MagicMock()
    stub.columns.side_effect = lambda spec: [MagicMock() for _ in (range(spec) if isinstance(spec, int) else spec)]
    stub.selectbox.side_effect = lambda label, options, **kw: list(options)[0]
    monkeypatch.setitem(__import__("sys").modules, "streamlit", stub)
    monkeypatch.setattr(__import__("sys"), "argv", ["dashboard.py", str(project)])
    monkeypatch.delitem(__import__("sys").modules, "mlguy.dashboard", raising=False)

    import mlguy.dashboard  # noqa: F401  (importing runs the page)

    assert stub.line_chart.called and stub.dataframe.called


def test_new_scaffolds_a_runnable_project(tmp_path):
    target = tmp_path / "fresh"
    assert main(["new", str(target), "--name", "fresh-project"]) == 0
    for name in ["task/task.toml", "task/brief.md", "infer.py", "proposal.md", "PROJECT_MEMORY.md"]:
        assert (target / name).exists(), name
    assert "fresh-project" in (target / "task" / "task.toml").read_text()
    assert main(["new", str(target), "--name", "again"]) == 2, "must not overwrite a project"


def test_validation_test_mismatch_is_carried_into_init_and_report(project, capsys):
    toml = project / "task" / "task.toml"
    toml.write_text(toml.read_text().replace(
        "[metric]",
        'validation_matches_test = false\n'
        'validation_note = "Validation is drawn like training; the test set is not."\n\n[metric]',
    ))
    run(project, "init")
    assert "does not ask the same question" in capsys.readouterr().out

    run(project, "validate")
    _write_model(project, 2.0)
    run(project, "submit", "--note", "double the scale")
    run(project, "finish")
    report = (project / "REPORT.md").read_text()
    assert "does not ask the same question" in report
    assert "Validation is drawn like training" in report


TABULAR_TOML = """\
schema_version = "0.1"
name = "tabular-test-project"

[task]
type = "regression"
input = "a cell line and a drug dose"
output = "response"
higher_is_better = true

[data]
""" + DATA_TOML + """
[metric]
primary = "pearson"

[budget]
max_submissions = 1
"""

TABULAR_PREDICT = '''\
import argparse, json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, required=True)
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()

w = json.loads((a.model / "weights.json").read_text())
rows = pd.read_csv(a.input)
assert list(rows.columns) == ["id", "cell_line", "dose", "replicate"], rows.columns
rows["prediction"] = rows["dose"] * w["slope"] + rows["cell_line"].map({"A": 1.0, "B": 2.0}).fillna(0)
rows[["id", "prediction"]].sample(frac=1, random_state=0).to_csv(a.output, index=False)   # any order
'''


def test_a_project_with_no_sequences_at_all(tmp_path, monkeypatch):
    """Several input columns, none of them a sequence: the core must not care."""
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "state"))
    project = tmp_path / "tabular"
    (project / "task" / "data").mkdir(parents=True)
    (project / "task" / "task.toml").write_text(TABULAR_TOML)

    rng = random.Random(1)
    rows = [{"cell_line": rng.choice("AB"), "dose": rng.random() * 10,
             "replicate": rng.randint(1, 3)} for _ in range(90)]
    for row in rows:
        row["target"] = row["dose"] * 1.5 + (1.0 if row["cell_line"] == "A" else 2.0) + rng.random()
    frame = pd.DataFrame(rows)
    write_splits(project / "task" / "data", frame[:50], frame[50:70], frame[70:])

    (project / "infer.py").write_text(TABULAR_PREDICT)
    (project / "model").mkdir()
    (project / "model" / "weights.json").write_text(json.dumps({"slope": 1.5}))

    assert run(project, "init") == 0
    hidden = pd.read_csv(project / "task" / "data" / "test_inputs.csv")
    assert list(hidden.columns) == ["id", "cell_line", "dose", "replicate"], "init leaves the inputs alone"
    assert run(project, "validate") == 0
    assert Experiments(project).best("pearson")["metrics"]["pearson"] > 0.9
    assert run(project, "finish") == 0
    assert "response" in (project / "REPORT.md").read_text()


def _tabular_project(tmp_path, monkeypatch):
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "state"))
    project = tmp_path / "tabular"
    (project / "task" / "data").mkdir(parents=True)
    (project / "task" / "task.toml").write_text(TABULAR_TOML)
    rng = random.Random(2)
    rows = [{"cell_line": rng.choice("AB"), "dose": rng.random() * 10,
             "replicate": rng.randint(1, 3)} for _ in range(60)]
    for row in rows:
        row["target"] = row["dose"] * 1.5 + rng.random()
    frame = pd.DataFrame(rows)
    write_splits(project / "task" / "data", frame[:30], frame[30:45], frame[45:])
    (project / "infer.py").write_text(TABULAR_PREDICT)
    (project / "model").mkdir()
    (project / "model" / "weights.json").write_text(json.dumps({"slope": 1.5}))
    run(project, "init")
    return project


def test_serve_runs_the_projects_own_page(tmp_path, monkeypatch, capfd):
    """The framework ships no page: `serve` runs whatever app.py the agent built."""
    project = _tabular_project(tmp_path, monkeypatch)
    assert run(project, "serve") == 2, "no app.py yet, so there is nothing to serve"
    assert "not been built" in capfd.readouterr().err

    (project / "app.py").write_text("import sys; print('page', *sys.argv[1:])\n")
    assert run(project, "serve", "--port", "7001", "--share") == 0
    assert "page --port 7001 --share" in capfd.readouterr().out


# ------------------------------------------------------------ the rubric cannot be bypassed


def test_the_test_set_cannot_be_scored_outside_finish(project):
    run(project, "init")
    with pytest.raises(SystemExit):
        run(project, "score", "--split", "test")
    assert run(project, "score") == 0, "validation scoring stays available"


def test_the_baseline_cannot_become_a_free_submission(project):
    run(project, "init")
    run(project, "validate")
    _write_model(project, 3.0)
    assert run(project, "validate", "--again") == 0, "fixing the baseline before any submission is fine"
    assert [r["number"] for r in Experiments(project).all()] == [0], "and it replaces the record"

    run(project, "submit", "--note", "a counted submission")
    assert run(project, "validate", "--again") == 2


def test_a_changed_task_cannot_be_refrozen_once_there_are_results(project, capsys):
    run(project, "init")
    toml = project / "task" / "task.toml"
    toml.write_text(toml.read_text().replace("max_submissions = 2", "max_submissions = 3"))
    assert run(project, "init") == 0, "still setting up: nothing has been scored against the old task"

    run(project, "validate")
    toml.write_text(toml.read_text().replace("max_submissions = 3", "max_submissions = 99"))
    assert run(project, "init") == 2
    assert "cannot be re-frozen" in capsys.readouterr().err


def test_scores_come_with_their_noise(project, capsys):
    run(project, "init")
    run(project, "validate")
    baseline = Experiments(project).all()[0]
    low, high = baseline["interval"]
    assert low < baseline["metrics"]["spearman"] < high

    _write_model(project, -1.0)
    run(project, "submit", "--note", "flip the sign")
    record = Experiments(project).all()[-1]
    assert record["vs_best"] == {"number": 0, "chance_better": 0.0}
    assert "really worse" in capsys.readouterr().out

    _write_model(project, 2.0)  # same ranking as the baseline
    run(project, "submit", "--note", "double the scale")
    assert Experiments(project).all()[-1]["vs_best"]["chance_better"] == 0.5
    assert "cannot be told apart" in capsys.readouterr().out

    run(project, "finish")
    assert "95% interval" in (project / "REPORT.md").read_text()


def test_init_flags_a_split_that_would_mislead(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "state"))
    project = tmp_path / "leaky"
    (project / "task" / "data").mkdir(parents=True)
    (project / "task" / "task.toml").write_text(
        TASK_TOML.replace('type = "regression"', 'type = "classification"')
                 .replace('primary = "spearman"', 'primary = "accuracy"').replace('secondary = ["ndcg"]', ""))
    rng = random.Random(3)
    seqs = ["".join(rng.choice("ACDE") for _ in range(8)) for _ in range(400)]
    frame = pd.DataFrame({"sequence": seqs}).drop_duplicates("sequence").reset_index(drop=True)
    train, validation, test = frame[:150].copy(), frame[150:270].copy(), frame[270:330].copy()
    train["target"] = [i % 2 for i in range(len(train))]
    validation["target"] = [1 if i % 5 == 0 else 0 for i in range(len(validation))]   # 20% positive
    test["target"] = [i % 2 for i in range(len(test))]                                 # 50% positive
    test = pd.concat([test, train[:7]])                                                # 7 leaked rows
    write_splits(project / "task" / "data", train, validation, test)

    assert run(project, "init") == 0, "checks warn; they do not refuse"
    out = capsys.readouterr().out
    assert "7 rows have identical inputs in train and test" in out
    assert "test has only 67 examples" in out
    assert "80% of validation but" in out and "balanced_accuracy" in out


FOLDER_INFER = '''\
import argparse, json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path); p.add_argument("--input", type=Path); p.add_argument("--output")
a = p.parse_args()
rows = [{"id": f.stem, "prediction": json.loads(f.read_text())["reading"] * 2} for f in sorted(a.input.glob("*.json"))]
pd.DataFrame(rows).to_csv(a.output, index=False)
'''


def test_inputs_can_be_anything_the_projects_code_can_read(tmp_path, monkeypatch):
    """One file per example in a folder, no table anywhere: the core only needs ids and labels."""
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "state"))
    project = tmp_path / "files"
    data = project / "task" / "data"
    toml = TABULAR_TOML.replace(DATA_TOML, 'train = "data/train"\nvalidation_inputs = "data/validation"\n'
                                'validation_labels = "data/validation_labels.csv"\ntest_inputs = "data/test"\n'
                                'test_labels = "data/test_labels.csv"\n')
    assert "data/validation\"" in toml
    data.mkdir(parents=True)
    (project / "task" / "task.toml").write_text(toml)
    rng = random.Random(4)
    for split, count in [("train", 20), ("validation", 120), ("test", 120)]:
        (data / split).mkdir()
        labels = []
        for i in range(count):
            reading = rng.random()
            (data / split / f"{split}-{i}.json").write_text(json.dumps({"reading": reading}))
            labels.append({"id": f"{split}-{i}", "target": reading * 2 + rng.random() * 0.1})
        if split != "train":
            pd.DataFrame(labels).to_csv(data / f"{split}_labels.csv", index=False)
    (project / "infer.py").write_text(FOLDER_INFER)
    (project / "model").mkdir()

    assert run(project, "init") == 0
    assert (data / "test").is_dir() and not (data / "test_labels.csv").exists()
    assert run(project, "validate") == 0
    assert Experiments(project).best("pearson")["metrics"]["pearson"] > 0.99
    assert run(project, "finish") == 0

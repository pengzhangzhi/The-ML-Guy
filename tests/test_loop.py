"""The loop end to end on a tiny made-up project: init, validate, submit, budget, finish."""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

import pandas as pd
import pytest

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
train = "data/train.csv"
validation = "data/validation.csv"
test = "data/test.csv"
test_inputs = "data/test_inputs.csv"
input_columns = ["sequence"]
target_column = "target"

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
seqs = pd.read_csv(a.input)["sequence"].tolist()
pd.DataFrame({{"prediction": [scale * s.count("A") + {offset} for s in seqs]}}).to_csv(a.output, index=False)
'''


def _write_model(project: Path, scale: float) -> None:
    (project / "model").mkdir(exist_ok=True)
    (project / "model" / "weights.json").write_text(json.dumps({"scale": scale}))


def _write_predict(project: Path, offset: str = "0") -> None:
    (project / "predict.py").write_text(PREDICT_PY.format(offset=offset))


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
    for name, part in [("train", frame[:60]), ("validation", frame[60:90]), ("test", frame[90:])]:
        part.to_csv(project / "task" / "data" / f"{name}.csv", index=False)

    _write_predict(project)
    _write_model(project, 1.0)
    return project


def run(project: Path, *args: str) -> int:
    return main([args[0], "--project", str(project), *args[1:]])


def test_full_loop(project, capsys):
    assert run(project, "init") == 0
    assert not (project / "task" / "data" / "test.csv").exists(), "labelled test file should be gone"
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
    assert run(project, "finish") == 2, "the test set must not be scored twice"


def test_broken_submission_is_an_error_not_a_zero(project, capsys):
    run(project, "init")
    run(project, "validate")
    (project / "predict.py").write_text("raise SystemExit('boom')")
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
    for name in ["task/task.toml", "task/brief.md", "predict.py", "src/train.py", "app.py",
                 "proposal.md", "PROJECT_MEMORY.md"]:
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
train = "data/train.csv"
validation = "data/validation.csv"
test = "data/test.csv"
test_inputs = "data/test_inputs.csv"
input_columns = ["cell_line", "dose", "replicate"]
target_column = "target"

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
assert list(rows.columns) == ["cell_line", "dose", "replicate"], rows.columns
pred = rows["dose"] * w["slope"] + rows["cell_line"].map({"A": 1.0, "B": 2.0}).fillna(0)
pd.DataFrame({"prediction": pred}).to_csv(a.output, index=False)
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
    for name, part in [("train", frame[:50]), ("validation", frame[50:70]), ("test", frame[70:])]:
        part.to_csv(project / "task" / "data" / f"{name}.csv", index=False)

    (project / "predict.py").write_text(TABULAR_PREDICT)
    (project / "model").mkdir()
    (project / "model" / "weights.json").write_text(json.dumps({"slope": 1.5}))

    assert run(project, "init") == 0
    hidden = pd.read_csv(project / "task" / "data" / "test_inputs.csv")
    assert list(hidden.columns) == ["cell_line", "dose", "replicate"], "init must keep every input column"
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
    for name, part in [("train", frame[:30]), ("validation", frame[30:45]), ("test", frame[45:])]:
        part.to_csv(project / "task" / "data" / f"{name}.csv", index=False)
    (project / "predict.py").write_text(TABULAR_PREDICT)
    (project / "model").mkdir()
    (project / "model" / "weights.json").write_text(json.dumps({"slope": 1.5}))
    run(project, "init")
    return project


def test_the_page_handles_a_project_with_no_sequences(tmp_path, monkeypatch):
    from mlguy.app import parse_pasted, predict_table, read_table
    from mlguy.project import load_task

    project = _tabular_project(tmp_path, monkeypatch)
    task = load_task(project)

    upload = tmp_path / "new.csv"
    pd.DataFrame({"cell_line": ["A", "B"], "dose": [1.0, 2.0], "replicate": [1, 1],
                  "note": ["ignored", "ignored"]}).to_csv(upload, index=False)
    result, rejected = predict_table(project, read_table(upload, task))
    assert list(result.columns) == ["rank", "id", "prediction", "cell_line", "dose", "replicate"]
    assert len(result) == 2 and len(rejected) == 0

    missing = tmp_path / "missing.csv"
    pd.DataFrame({"cell_line": ["A"]}).to_csv(missing, index=False)
    with pytest.raises(ValueError, match="dose"):
        read_table(missing, task)

    with pytest.raises(ValueError, match="upload a CSV"):
        parse_pasted("A,1.0,1", task)


def test_the_page_rejects_rows_with_missing_values(tmp_path, monkeypatch):
    from mlguy.app import predict_table

    project = _tabular_project(tmp_path, monkeypatch)
    table = pd.DataFrame({"cell_line": ["A", None], "dose": [1.0, 2.0], "replicate": [1, 1]})
    result, rejected = predict_table(project, table)
    assert len(result) == 1 and len(rejected) == 1
    assert "missing" in rejected.iloc[0]["reason"]

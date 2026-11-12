"""A task's own evaluator, and generation: the same loop with a different way of producing and scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from helpers import DATA_TOML, write_splits, write_task
from mlguy.cli import main
from mlguy.records import Experiments

# A toy conditional generator: for each group, draw numbers that should land near that group's
# real mean. The evaluator scores each sample by how far it falls from the reference mean.
GEN_TOML = """\
name = "toy-generation"

[task]
type = "generation"
input = "a group"
output = "measurements"

[data]
""" + DATA_TOML + """
[sampling]
n = 20
seed = 0
final_n = 50

[evaluator]
script = "evaluate.py"

[metric]
primary = "distance"
direction = "minimize"

[budget]
max_submissions = 2
"""

EVALUATE_PY = '''\
import argparse, json
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--outputs"); p.add_argument("--reference"); p.add_argument("--inputs"); p.add_argument("--split"); p.add_argument("--result")
a = p.parse_args()

samples, truth = pd.read_csv(a.outputs), pd.read_csv(a.reference)
if "value" not in samples:
    raise SystemExit("samples need a 'value' column")
distance = (samples["value"] - samples["id"].map(truth.set_index("id")["mean"])).abs()
json.dump({"metrics": {"distance": float(distance.mean()), "spread": float(samples["value"].std()),
                       "final": float(a.split == "final")}}, open(a.result, "w"))
'''

GENERATE_PY = '''\
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path); p.add_argument("--input"); p.add_argument("--n", type=int)
p.add_argument("--seed", type=int); p.add_argument("--output")
a = p.parse_args()

means = json.loads((a.model / "means.json").read_text())
rng = np.random.default_rng(a.seed)
rows = [{"id": c.id, "value": means.get(c.group, 0.0) + rng.normal(scale=0.1)}
        for c in pd.read_csv(a.input).itertuples() for _ in range(a.n)]
pd.DataFrame(rows)%s.to_csv(a.output, index=False)
'''

TRUE_MEANS = {"a": 1.0, "b": 5.0, "c": 9.0}


def run(project: Path, *args: str) -> int:
    return main([args[0], "--project", str(project), *args[1:]])


def _model(project: Path, error: float) -> None:
    (project / "model").mkdir(exist_ok=True)
    (project / "model" / "means.json").write_text(json.dumps({g: m + error for g, m in TRUE_MEANS.items()}))


@pytest.fixture()
def generation(tmp_path, monkeypatch):
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "state"))
    project = tmp_path / "gen"
    (project / "task" / "data").mkdir(parents=True)
    write_task(project, GEN_TOML)
    (project / "task" / "evaluate.py").write_text(EVALUATE_PY)
    reference = pd.DataFrame({"group": list(TRUE_MEANS), "mean": list(TRUE_MEANS.values())})
    write_splits(project / "task" / "data", pd.DataFrame({"group": ["a", "b", "c"] * 10, "value": [1.0, 5.0, 9.0] * 10}),
                 reference, reference.iloc[::-1], target="mean")
    (project / "infer.py").write_text(GENERATE_PY % "")
    _model(project, error=2.0)
    return project


def test_generation_loop(generation, capsys):
    project = generation
    assert run(project, "init") == 0
    out = capsys.readouterr().out
    assert "20 per condition on every submission" in out and "50 per condition for the final" in out
    data = project / "task" / "data"
    assert not (data / "test_labels.csv").exists() and not (data / "test_inputs.csv").exists(), \
        "the final conditions stay unseen until finish"

    assert run(project, "validate") == 0
    experiments = Experiments(project)
    baseline = experiments.all()[0]
    assert baseline["metrics"]["distance"] == pytest.approx(2.0, abs=0.1)
    assert baseline["n_validation"] == 60
    assert list(baseline["metrics"]) == ["distance", "spread", "final"], "primary first, then whatever else it reports"
    assert baseline["metrics"]["final"] == 0.0

    _model(project, error=0.5)
    assert run(project, "submit", "--note", "closer means") == 0
    assert experiments.best("distance", "minimize")["number"] == 1
    assert (experiments.dir / "0001" / "submission" / "infer.py").exists()

    assert run(project, "finish") == 0
    final = json.loads((project.parent / "state/projects/toy-generation/test_score.json").read_text())
    assert final["number"] == 1 and final["n"] == 150, "50 fresh samples for each held-out condition"
    assert final["metrics"]["final"] == 1.0, "the evaluator is told this is the final check"
    report = (project / "REPORT.md").read_text()
    assert "Generates **measurements**" in report and "--n 100" in report
    assert run(project, "finish") == 2


def test_malformed_samples_are_an_error_not_a_score(generation):
    project = generation
    run(project, "init")
    run(project, "validate")
    (project / "infer.py").write_text(GENERATE_PY % ".iloc[:-3]")     # three samples short
    assert run(project, "submit", "--note", "drop some") == 1
    assert "20 samples for each of 3 conditions" in Experiments(project).all()[-1]["error"]

    (project / "infer.py").write_text(GENERATE_PY % '.rename(columns={"value": "v"})')
    assert run(project, "submit", "--note", "wrong column") == 1
    assert "samples need a 'value' column" in Experiments(project).all()[-1]["error"], \
        "the evaluator's own complaint reaches the agent"


def test_an_evaluator_can_be_tried_on_a_file(generation, capsys, tmp_path):
    """Before trusting an evaluator, feed it something good and something bad."""
    project = generation
    run(project, "init")
    capsys.readouterr()
    for name, offset in [("good", 0.0), ("bad", 4.0)]:
        pd.DataFrame({"id": ["v0", "v1", "v2"], "value": [1.0 + offset, 5.0 + offset, 9.0 + offset]}).to_csv(
            tmp_path / f"{name}.csv", index=False)
    scores = {}
    for name in ("good", "bad"):
        assert run(project, "score", "--outputs", str(tmp_path / f"{name}.csv")) == 0
        scores[name] = json.loads(capsys.readouterr().out)["metrics"]["distance"]
    assert scores["good"] == 0.0 and scores["bad"] == 4.0
    assert Experiments(project).all() == [], "nothing is recorded"


def test_the_evaluator_is_frozen(generation, capsys):
    project = generation
    run(project, "init")
    run(project, "validate")
    script = project / "task" / "evaluate.py"
    script.write_text(script.read_text() + "\n# make it easier\n")
    assert run(project, "submit", "--note", "after editing the evaluator") == 2
    assert "task/ has changed" in capsys.readouterr().err


PRED_TOML = """\
name = "toy-custom-metric"

[task]
type = "regression"
input = "a number"
output = "another_number"

[data]
""" + DATA_TOML + """target_column = "y"

[evaluator]
script = "evaluate.py"

[metric]
primary = "worst_error"
direction = "minimize"
"""

PRED_EVALUATE = '''\
import argparse, json
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--outputs"); p.add_argument("--reference"); p.add_argument("--inputs"); p.add_argument("--split"); p.add_argument("--result")
a = p.parse_args()
both = pd.read_csv(a.outputs).merge(pd.read_csv(a.reference), on="id")
error = (both["prediction"] - both["y"]).abs()
json.dump({"metrics": {"worst_error": float(error.max())}}, open(a.result, "w"))
'''

PRED_PREDICT = '''\
import argparse, json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path); p.add_argument("--input"); p.add_argument("--output")
a = p.parse_args()
slope = json.loads((a.model / "w.json").read_text())["slope"]
rows = pd.read_csv(a.input)
rows.assign(prediction=rows["x"] * slope)[["id", "prediction"]].to_csv(a.output, index=False)
'''


def test_a_prediction_task_can_bring_its_own_metric(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "state"))
    project = tmp_path / "custom"
    (project / "task" / "data").mkdir(parents=True)
    write_task(project, PRED_TOML)
    (project / "task" / "evaluate.py").write_text(PRED_EVALUATE)
    frame = pd.DataFrame({"x": [float(i) for i in range(300)]})
    frame["y"] = frame["x"] * 2
    write_splits(project / "task" / "data", frame[:100], frame[100:200], frame[200:], target="y")
    (project / "infer.py").write_text(PRED_PREDICT)
    (project / "model").mkdir()
    (project / "model" / "w.json").write_text(json.dumps({"slope": 1.0}))

    assert run(project, "init") == 0
    assert list(pd.read_csv(project / "task" / "data" / "test_inputs.csv").columns) == ["id", "x"]
    assert run(project, "validate") == 0
    (project / "model" / "w.json").write_text(json.dumps({"slope": 2.0}))
    assert run(project, "submit", "--note", "the right slope") == 0
    assert Experiments(project).best("worst_error", "minimize")["number"] == 1
    assert run(project, "finish") == 0
    assert "worst_error = 0.0000" in capsys.readouterr().out


UNDEFINED_SECONDARY = PRED_EVALUATE.replace(
    'json.dump({"metrics": {"worst_error": float(error.max())}}, open(a.result, "w"))',
    'json.dump({"metrics": {"worst_error": float(error.max()), "ratio": float("nan")}}, open(a.result, "w"))')


def test_only_the_primary_metric_must_be_a_number(tmp_path, monkeypatch, capsys):
    """A secondary metric can be undefined for some outputs; that must not void the submission."""
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "state"))
    project = tmp_path / "custom"
    (project / "task" / "data").mkdir(parents=True)
    write_task(project, PRED_TOML)
    (project / "task" / "evaluate.py").write_text(UNDEFINED_SECONDARY)
    frame = pd.DataFrame({"x": [float(i) for i in range(300)]})
    frame["y"] = frame["x"] * 2
    write_splits(project / "task" / "data", frame[:100], frame[100:200], frame[200:], target="y")
    (project / "infer.py").write_text(PRED_PREDICT)
    (project / "model").mkdir()
    (project / "model" / "w.json").write_text(json.dumps({"slope": 1.0}))
    run(project, "init")
    assert run(project, "validate") == 0
    assert Experiments(project).all()[0]["metrics"]["worst_error"] == pytest.approx(199.0)


MACRO_EVALUATE = '''\
import argparse, json
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--outputs"); p.add_argument("--reference"); p.add_argument("--inputs"); p.add_argument("--split"); p.add_argument("--result")
a = p.parse_args()
both = pd.read_csv(a.outputs).merge(pd.read_csv(a.reference), on="id")
both["right"] = both["prediction"] == both["y"]
json.dump({"metrics": {"macro_recall": float(both.groupby("y")["right"].mean().mean())}}, open(a.result, "w"))
'''

LABEL_INFER = '''\
import argparse, json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path); p.add_argument("--input"); p.add_argument("--output")
a = p.parse_args()
cut = json.loads((a.model / "w.json").read_text())["cut"]
rows = pd.read_csv(a.input)
rows.assign(prediction=(rows["x"] > cut).map({True: "high", False: "low"}))[["id", "prediction"]].to_csv(a.output, index=False)
'''


def test_an_evaluator_for_class_labels(tmp_path, monkeypatch, capsys):
    """A per-class average is not a built-in; the labels are still checked at init."""
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "state"))
    project = tmp_path / "macro"
    (project / "task" / "data").mkdir(parents=True)
    toml = PRED_TOML.replace('primary = "worst_error"\ndirection = "minimize"', 'primary = "macro_recall"\ndirection = "maximize"')
    write_task(project, toml.replace('type = "regression"', 'type = "classification"'))
    (project / "task" / "evaluate.py").write_text(MACRO_EVALUATE)
    frame = pd.DataFrame({"x": [float(i % 100) for i in range(300)]})
    frame["y"] = (frame["x"] > 70).map({True: "high", False: "low"})
    write_splits(project / "task" / "data", frame[:100], frame[100:200], frame[200:], target="y")
    (project / "infer.py").write_text(LABEL_INFER)
    (project / "model").mkdir()
    (project / "model" / "w.json").write_text(json.dumps({"cut": 50}))

    assert run(project, "init") == 0
    assert "not compared" not in capsys.readouterr().out, "labels are compared even with an evaluator"
    assert run(project, "validate") == 0
    (project / "model" / "w.json").write_text(json.dumps({"cut": 70}))
    assert run(project, "submit", "--note", "the right cut") == 0
    assert Experiments(project).best("macro_recall", "maximize")["number"] == 1
    assert (project / "experiments" / "0001" / "validation_outputs.csv").exists()


def test_finish_can_take_a_simpler_submission_with_a_reason(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "state"))
    project = tmp_path / "custom"
    (project / "task" / "data").mkdir(parents=True)
    write_task(project, PRED_TOML)
    (project / "task" / "evaluate.py").write_text(PRED_EVALUATE)
    frame = pd.DataFrame({"x": [float(i) for i in range(300)]})
    frame["y"] = frame["x"] * 2
    write_splits(project / "task" / "data", frame[:100], frame[100:200], frame[200:], target="y")
    (project / "infer.py").write_text(PRED_PREDICT)
    (project / "model").mkdir()
    (project / "model" / "w.json").write_text(json.dumps({"slope": 1.9}))
    run(project, "init")
    run(project, "validate")
    (project / "model" / "w.json").write_text(json.dumps({"slope": 2.0}))
    run(project, "submit", "--note", "exact")
    capsys.readouterr()

    assert run(project, "finish", "--experiment", "0") == 2, "a reason is required"
    assert "--reason" in capsys.readouterr().err
    assert run(project, "finish", "--experiment", "0", "--reason", "simpler and within noise") == 0
    assert json.loads((project / "final" / "model" / "w.json").read_text()) == {"slope": 1.9}
    assert "Why: simpler and within noise" in (project / "REPORT.md").read_text()


def test_the_state_location_is_remembered(tmp_path, monkeypatch):
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "elsewhere"))
    project = tmp_path / "custom"
    (project / "task" / "data").mkdir(parents=True)
    write_task(project, PRED_TOML)
    (project / "task" / "evaluate.py").write_text(PRED_EVALUATE)
    frame = pd.DataFrame({"x": [float(i) for i in range(300)]})
    frame["y"] = frame["x"] * 2
    write_splits(project / "task" / "data", frame[:100], frame[100:200], frame[200:], target="y")
    (project / "infer.py").write_text(PRED_PREDICT)
    (project / "model").mkdir()
    (project / "model" / "w.json").write_text(json.dumps({"slope": 2.0}))
    assert run(project, "init") == 0
    monkeypatch.delenv("MLGUY_HOME")
    assert run(project, "validate") == 0 and run(project, "finish") == 0, "found without MLGUY_HOME"

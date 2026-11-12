"""The loop end to end on a tiny made-up project: init, validate, submit, budget, finish."""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from helpers import DATA_TOML, write_splits, write_task
from mlguy.cli import main
from mlguy.records import Experiments

TASK_TOML = """\
schema_version = "0.1"
name = "tiny-test-project"

[task]
type = "regression"
input = "protein_sequence"
output = "activity"

[input]
alphabet = "ACDE"
length = 8

[data]
""" + DATA_TOML + """
[metric]
primary = "spearman"
direction = "maximize"

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
    write_task(project, TASK_TOML)

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
    assert experiments.best("spearman", "maximize")["number"] == 0

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
    assert Experiments(project).best("spearman", "maximize")["number"] == 0


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


def test_a_model_that_reads_task_fails_at_validate_not_at_finish(project, capsys):
    """Submissions are scored from their own copy, which holds no task/.

    A model that reads its data out of task/ when it predicts used to pass every submission and
    fail at `mlguy finish`, with the budget spent and the test set about to be opened. It has to
    fail on the free baseline instead, and the error has to say why.
    """
    (project / "task" / "data" / "offset.txt").write_text("1.5")
    run(project, "init")
    (project / "infer.py").write_text(
        'import argparse, json\n'
        'from pathlib import Path\n'
        'import pandas as pd\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("--model", type=Path, required=True)\n'
        'p.add_argument("--input", type=Path, required=True)\n'
        'p.add_argument("--output", type=Path, required=True)\n'
        'a = p.parse_args()\n'
        '# the trap: a reference file read relative to this script, inside task/\n'
        'offset = float((Path(__file__).resolve().parent / "task" / "data" / "offset.txt").read_text())\n'
        'rows = pd.read_csv(a.input)\n'
        'rows["prediction"] = [offset + s.count("A") for s in rows["sequence"]]\n'
        'rows[["id", "prediction"]].to_csv(a.output, index=False)\n'
    )

    assert run(project, "validate") == 1
    message = capsys.readouterr().err
    assert "task/" in message and "model/" in message
    # nothing was recorded, so the loop never opened on a pipeline that cannot be reproduced
    assert Experiments(project).all() == []


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


def test_new_scaffolds_a_runnable_project(tmp_path):
    target = tmp_path / "fresh"
    assert main(["new", str(target), "--name", "fresh-project"]) == 0
    for name in ["task/task.toml", "task/brief.md", "task/evaluate.py", "infer.py", "PROJECT_MEMORY.md"]:
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

[data]
""" + DATA_TOML + """
[metric]
primary = "pearson"
direction = "maximize"

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
    write_task(project, TABULAR_TOML)

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
    assert Experiments(project).best("pearson", "maximize")["metrics"]["pearson"] > 0.9
    assert run(project, "finish") == 0
    assert "response" in (project / "REPORT.md").read_text()


def _tabular_project(tmp_path, monkeypatch):
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "state"))
    project = tmp_path / "tabular"
    (project / "task" / "data").mkdir(parents=True)
    write_task(project, TABULAR_TOML)
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


# ------------------------------------------------------------ the rubric cannot be bypassed


def test_score_only_checks_a_given_file(project, tmp_path, capsys):
    """Neither split can be scored by running the model outside submit and finish."""
    run(project, "init")
    for args in (["score"], ["score", "--split", "test"]):
        with pytest.raises(SystemExit):
            run(project, *args)
    ids = pd.read_csv(project / "task" / "data" / "validation_labels.csv")["id"]
    pd.DataFrame({"id": ids, "prediction": range(len(ids))}).to_csv(tmp_path / "out.csv", index=False)
    capsys.readouterr()
    assert run(project, "score", "--outputs", str(tmp_path / "out.csv")) == 0
    assert "spearman" in capsys.readouterr().out
    assert Experiments(project).all() == [], "nothing is recorded"


def test_finish_leaves_the_test_outputs_and_labels_in_final(project):
    run(project, "init")
    run(project, "validate")
    run(project, "finish")
    outputs = pd.read_csv(project / "final" / "test_outputs.csv")
    labels = pd.read_csv(project / "final" / "test_labels.csv")
    assert set(outputs["id"]) == set(labels["id"]) and "target" in labels
    report = (project / "REPORT.md").read_text()
    assert "0 of 2 submissions used" in report and "after the budget was spent" not in report
    assert "(higher is better)" in report


def test_python_caches_in_task_do_not_count_as_a_change(project, capsys):
    """Importing task/evaluate.py from training code makes Python write task/__pycache__."""
    run(project, "init")
    run(project, "validate")
    cache = project / "task" / "__pycache__"
    cache.mkdir()
    (cache / "evaluate.cpython-311.pyc").write_bytes(b"\x00")
    _write_model(project, 2.0)
    assert run(project, "submit", "--note", "after a cache appeared") == 0

    (project / "task" / "data" / "validation_labels.csv").write_text("id,target\nv0,1\n")
    assert run(project, "submit", "--note", "after the labels changed") == 2
    assert "data/validation_labels.csv (changed)" in capsys.readouterr().err, "the error names the file"


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


def test_each_submission_is_one_score_set_against_the_best(project, capsys):
    run(project, "init")
    run(project, "validate")
    capsys.readouterr()

    _write_model(project, -1.0)
    run(project, "submit", "--note", "flip the sign")
    record = Experiments(project).all()[-1]
    assert set(record) >= {"metrics", "n_validation"} and "interval" not in record
    out = capsys.readouterr().out
    assert "-1.9397 spearman against experiment 0, the best before this" in out
    assert "best so far: experiment 0" in out


def test_what_a_submission_costs_to_run_is_recorded_and_shown(project, capsys):
    """Selection is the agent's judgement, but it cannot weigh cost against score unless the cost
    is there to read — and the timing cannot be recovered once the run is over."""
    run(project, "init")
    run(project, "validate")
    _write_model(project, -1.0)
    run(project, "submit", "--note", "flip the sign")
    capsys.readouterr()

    for record in Experiments(project).submissions():
        cost = record["cost"]
        assert cost["predict_seconds"] >= 0, "how long infer.py took over validation"
        assert cost["model_mb"] >= 0, "and how big model/ is"

    assert run(project, "status") == 0
    out = capsys.readouterr().out
    assert "cost: seconds to run the model over validation" in out
    assert "MB" in out

    capsys.readouterr()
    assert run(project, "finish") == 0
    report = (project / "REPORT.md").read_text()
    assert "on disk" in report and "validation examples on the machine" in report


def test_an_error_before_the_snapshot_exists_is_still_recorded(project, capsys):
    """Recording the cost must not make a failed submission crash instead of being recorded."""
    run(project, "init")
    run(project, "validate")
    (project / "infer.py").unlink()
    capsys.readouterr()

    assert run(project, "submit", "--note", "no infer.py at all") == 1
    record = Experiments(project).all()[-1]
    assert record["status"] == "error" and "cost" not in record
    assert record["counts_against_budget"], "a failure still uses up a submission"


def test_the_score_table_keeps_every_metric_the_evaluator_reports():
    """The experiment table truncates to stay readable; the score table must not, or a metric that
    disagrees with the headline can go unseen. One trial's evaluator reported nine and four showed."""
    from mlguy.report import MAX_COLUMNS, _all_metrics
    test = {"metrics": {f"m{i}": 0.0 for i in range(6)}}
    best = {"metrics": {"m0": 0.0, "only_scored_on_validation": 0.0}}
    names = _all_metrics("m3", best, test)
    assert names[0] == "m3", "the primary comes first"
    assert "only_scored_on_validation" in names, "a metric one side alone reports still has a row"
    assert len(names) == 7 and len(names) > MAX_COLUMNS


def test_the_report_and_status_name_the_result_rather_than_calling_it_the_best(project, capsys):
    """`finish --experiment N` takes a submission that did not win on validation, so every line
    that calls the result "the best submission" is then wrong — and after the final score has been
    taken, `status` has to say so instead of sending the agent back to `finish`."""
    run(project, "init")
    run(project, "validate")
    _write_model(project, -1.0)
    run(project, "submit", "--note", "flip the sign")
    _write_model(project, -0.5)
    run(project, "submit", "--note", "flip it less")
    capsys.readouterr()

    assert run(project, "finish", "--experiment", "1", "--reason", "the brief wants the sign") == 0
    report = (project / "REPORT.md").read_text()
    assert "best submission" not in report, "experiment 1 is the result; experiment 0 scored best"
    assert "Validation (experiment 1)" in report
    assert "experiment 1, the result, scored" in report
    assert "chosen over experiment 0" in report

    capsys.readouterr()
    assert run(project, "status") == 0
    out = capsys.readouterr().out
    assert "final: experiment 1 scored spearman=" in out and "on the test set, once" in out
    assert "Run `mlguy finish`" not in out, "it has run; sending an agent back to it is a dead end"
    assert "see REPORT.md" in out


def test_a_scoring_change_mid_project_says_how_to_check_it(project, capsys, monkeypatch):
    """mlguy cannot tell a change that moved a number from a refactor around it, so the warning
    has to point at the way to find out — and not at anything that costs a submission."""
    run(project, "init")
    run(project, "validate")
    baseline = Experiments(project).all()[0]
    capsys.readouterr()

    monkeypatch.setattr("mlguy.cli.scoring_digest", lambda: "0" * 12)
    assert run(project, "submit", "--note", "after an upgrade") == 0
    out = capsys.readouterr().out
    assert "scoring code changed during this project" in out
    assert "mlguy score --outputs experiments/0000/validation_outputs.csv" in out
    assert f"spearman={baseline['metrics']['spearman']:.4f}" in out, \
        "with the number to compare against, so the check is one command and one glance"
    assert (project / "experiments" / "0000" / "validation_outputs.csv").exists()


def test_init_flags_a_split_that_would_mislead(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLGUY_HOME", str(tmp_path / "state"))
    project = tmp_path / "leaky"
    (project / "task" / "data").mkdir(parents=True)
    write_task(project, 
        TASK_TOML.replace('type = "regression"', 'type = "classification"')
                 .replace('primary = "spearman"', 'primary = "accuracy"'))
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
    assert "80% of validation but" in out and "plain accuracy" in out


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
    write_task(project, toml)
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
    assert Experiments(project).best("pearson", "maximize")["metrics"]["pearson"] > 0.99
    assert run(project, "finish") == 0


def test_two_projects_cannot_share_a_name(project, tmp_path, capsys):
    import shutil

    other = tmp_path / "copy"
    shutil.copytree(project, other)
    assert run(project, "init") == 0
    assert run(other, "init") == 2
    assert "different name" in capsys.readouterr().err


def test_init_needs_the_users_approval_and_an_evaluator(project, capsys):
    brief = project / "task" / "brief.md"
    brief.write_text("# Brief\n\nApproved by the user: TODO their words, and the date.\n")
    assert run(project, "init") == 2
    assert "does not record the user's approval" in capsys.readouterr().err
    brief.write_text("# Brief\n\nApproved by the user: \"looks right, go ahead\" (2026-11-11)\n")
    (project / "task" / "evaluate.py").unlink()
    assert run(project, "init") == 2
    assert "no evaluator" in capsys.readouterr().err


def test_a_large_task_is_fingerprinted_without_reading_all_of_it(project, capsys):
    """Freezing must not cost a full read of the data on every submission, but a change
    anywhere in it must still be caught."""
    from mlguy.records import CROWDED, SMALL_FILE, task_manifest

    big = project / "task" / "data" / "big.bin"
    big.write_bytes(b"\0" * (SMALL_FILE + 4 * (1 << 20)))
    crowd = project / "task" / "data" / "many"
    crowd.mkdir()
    for i in range(CROWDED + 1):
        (crowd / f"{i}.txt").write_text(str(i))

    run(project, "init")
    run(project, "validate")
    present = sum(1 for p in (project / "task" / "data").rglob("*") if p.is_file())
    assert task_manifest(project)["data/"].startswith(f"{present} files, ")
    assert not any(key.startswith("data/") and key != "data/" for key in task_manifest(project)), \
        "the whole folder is one entry, so no file in it is read"

    # the middle of a large file: changed length is what is noticed
    content = bytearray(big.read_bytes())
    content[len(content) // 2:len(content) // 2] = b"x"
    big.write_bytes(bytes(content))
    assert run(project, "submit", "--note", "after growing the data") == 2
    assert "task/ has changed" in capsys.readouterr().err

    big.write_bytes(bytes(content[:len(content) // 2] + content[len(content) // 2 + 1:]))
    assert run(project, "submit", "--note", "data restored") == 0
    (crowd / "0.txt").write_text("changed, and a different length")
    assert run(project, "submit", "--note", "after editing one of many files") == 2
    assert "task/ has changed" in capsys.readouterr().err

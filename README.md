# The-ML-Guy

Describe a prediction problem to a coding agent (Claude Code or Codex) and get back a trained
model, a report, and a web page for running predictions. No ML background needed.

The design rule: **be exact about the problem and the scoring, and leave the modelling open.**

| Strict: fixed when you approve the brief | Free: the agent's call |
| --- | --- |
| Task: input, output, regression or classification | Model family and architecture |
| Data: train, validation and test sets | Features and encodings |
| Metric and its direction | Hyperparameters and training procedure |
| Budget | Code structure, libraries, order of work |
| Prediction interface: a table in, a prediction per row out | What to try next |

## Philosophy

The-ML-Guy is a thin layer on top of a coding agent, not a framework that runs one.

- **Minimal, but load-bearing.** It supplies only what an agent cannot be trusted or relied on
  to do for itself: an honest score, a budget that actually stops, state that survives the
  session, and a page other people can use. Everything else is left alone.
- **Maximum room for the agent.** The task, the data, the metric and the budget are fixed. The
  modelling is not: any architecture, any library, any code structure. Autoresearch is only as
  good as the freedom it has, so the tool stays out of it.
- **Domain-agnostic.** A project is a table of input columns and a target column. The core
  knows nothing about proteins, cells, patients or images. Sequence conveniences exist, but a
  project switches them on rather than inheriting them.
- **Agent-agnostic.** The contract is a command line and files on disk, so Claude Code, Codex
  or anything else can drive it, and a fresh session can pick up where the last one stopped.
- **No safety theatre.** Coding agents already carry their own controls. The-ML-Guy adds no
  sandbox and no permission layer; it protects scores from accidents, and says so plainly.

## Status

Early but complete end to end. All four milestones are in place: the project contract, the
scoring rubric, the hill-climbing loop with its budget and experiment records, the dashboard,
the prediction page, the three skills, and two worked examples from different problem shapes
(a variant landscape with a continuous target, and binary classification over unrelated
proteins of varying length).

See [INSTALL.md](INSTALL.md) to set it up for Claude Code or Codex.

There is no sandbox. The agent runs with your permissions. Scores are protected against
accidents (test labels live outside the project folder), not against an agent that
deliberately goes looking for them.

## Install

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[app,examples,dev]"
```

## Metrics

| Regression | Classification |
| --- | --- |
| `spearman`, `pearson`, `ndcg`, `rmse`, `mae` | `accuracy`, `balanced_accuracy`, `f1`, `mcc`, `roc_auc` |

A predictions file has one row per input row, in the same order, with a `prediction` column:
the number for regression, the class label for classification. Classification projects may add
a `score` column holding the probability of the positive class, which `roc_auc` needs and the
prediction page ranks by.

Only supervised prediction is supported today. Generative tasks are scored a different way —
by an external model or a problem-specific check rather than a held-out answer key — and need
a contract of their own, which is not designed yet.

## Example: amylase

One-hot ridge regression on the [FLIP2](https://flip.protein.properties/) alpha-amylase
one-to-many split.

```bash
cd examples/amylase
python make_task.py                 # download the split, write task/data
mlguy init                          # check the task, hide the test labels, freeze task/
python src/train.py                 # fit the baseline into model/
mlguy validate                      # check prediction and scoring work; records experiment 0

# the loop: change the modelling however you like, then
python src/train.py --approach mutation-ridge
mlguy submit --note "ridge on substitution indicators instead of full-sequence one-hot"
mlguy status                        # what has been tried, what is best, what budget is left
mlguy dashboard                     # the same, in a browser

mlguy finish                        # scores the best submission on test once, writes REPORT.md
mlguy serve                         # prediction page: upload CSV or FASTA, get ranked predictions
```

`mlguy submit` scores `model/` on validation, keeps a snapshot of the code and model that
produced the score, and refuses once the budget is spent. The test set is scored exactly once,
by `mlguy finish`, on whichever submission scored best on validation.

The split as published trains on the wild type and variants with 2 to 8 mutations, and tests
on the 488 single mutants. A run with a budget of three submissions:

| # | What was tried | Validation Spearman |
| --- | --- | --- |
| 0 | one-hot ridge over all 425 positions (baseline) | 0.6777 |
| 1 | ridge on substitution indicators plus mutation count | **0.6935** |
| 2 | gradient-boosted trees on the same features | 0.2702 |
| 3 | rank-average of the two ridge models | 0.6933 |

Experiment 1 won on validation and scored **0.6310** on the test set.

**Validation gains did not transfer.** A separately measured one-hot ridge scored 0.6964 on the
test set, better than the model that won on validation. This is what the split predicts:
validation is multi-mutants, like training, while the test set is single mutants. Climbing
validation here partly optimises the wrong question. Any project where validation and test ask
different questions has this problem, and the brief should say so.

NDCG here is over the full ranking with linear gain and targets shifted to start at zero. It
has not been checked against FLIP2's own definition.

## Example: solubility

Binary classification over 71,000 unrelated proteins of varying length
([DeepSol](https://huggingface.co/datasets/proteinea/solubility), MIT), a deliberately
different shape from amylase: no fixed length, no reference sequence, no position that means
the same thing twice.

```bash
cd examples/solubility
python make_task.py && mlguy init && python src/train.py && mlguy validate
```

Three submissions, CPU only, about four minutes of training in total:

| # | What was tried | Validation accuracy |
| --- | --- | --- |
| 0 | Logistic regression on residue composition (baseline) | 0.6263 |
| 1 | Add adjacent-pair (dipeptide) composition, 421 features | 0.6878 |
| 2 | Gradient-boosted trees on the same features | 0.6949 |
| 3 | Larger, slower-learning trees | **0.6995** |

Test: accuracy 0.6627, balanced accuracy 0.6626, MCC 0.338, ROC AUC 0.766. For context, the
PEER benchmark reports roughly 0.64 for a CNN and 0.70 for ESM-1b on this split, so simple
counting features on a CPU land within a few points of a pretrained protein language model.
Those comparison numbers are quoted from a secondary summary and have not been checked against
the paper.

This project declares `validation_matches_test = false`: training and validation are 42%
soluble while the test set is balanced 50/50, so plain accuracy is not comparable between
them. `PROJECT_MEMORY.md` records why the rubric cannot select the class-balanced model that
would probably score better on test, and what to set differently next time.

## Project layout

```
project/
├── task/            STRICT: brief.md, task.toml, data (train, validation, test inputs)
├── predict.py       FIXED INTERFACE: --model DIR --input FILE --output FILE
├── src/             FREE: any code, any structure
├── model/           the model as it stands now
├── experiments/     written only by `mlguy submit`: log.jsonl + a snapshot per submission
├── REPORT.md        written by `mlguy finish`
└── app.py           prediction page
```

Test labels live in `~/.mlguy/projects/<name>/`, outside the project folder, along with the
hash of `task/` taken when the brief was approved.

`predict.py` reads a CSV holding the task's input columns and writes a CSV with one
`prediction` per input row, in the same order. It is the only agent-written code with a fixed
shape, because both the rubric and the prediction page call it.

The input columns are whatever the problem needs: a protein sequence, a cell type and a dose,
a set of clinical measurements, paths to image files. The core treats them as opaque and hands
them to `predict.py` untouched.

```toml
[data]
input_columns = ["cell_line", "dose"]
target_column = "response"
```

A project whose input is a biological sequence can switch on two conveniences, which are off
by default and live in an optional `mlguy.sequence` module:

```toml
[page]
fasta_column = "sequence"      # also accept FASTA uploads and pasted sequences
mutations_from = "sequence"    # name mutations (Q68S) against [input].reference
```

## For agents

The `skills/` folder holds what a coding agent loads. Each skill states the goal, the rules and
the commands, and leaves the judgment to the agent.

| Skill | What it covers |
| --- | --- |
| `build-project` | Look at the data, ask only what is missing, write the brief, get it approved, scaffold and validate |
| `research` | The loop: the goal, the rubric, the budget, the commands. The approach is the agent's own |
| `deploy` | Score the test set once, write the report and the page in the user's language, record what was learned |

Starting a project by hand:

```bash
mlguy new my-project --name my-project   # skeleton: task.toml, brief.md, predict.py, a baseline
```

## Licence

MIT.

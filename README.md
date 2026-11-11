# The-ML-Guy

Describe a prediction problem to a coding agent (Claude Code or Codex) and get back a trained
model, a report, and a web page for running predictions. No ML background needed.

The design rule: **be exact about the problem and the scoring, and leave the modeling open.**

| Strict: fixed when you approve the brief | Free: the agent's call |
| --- | --- |
| Task: input, output, regression or classification | Model family and architecture |
| Data: train, validation and test sets | Features and encodings |
| Metric and its direction | Hyperparameters and training procedure |
| Budget | Code structure, libraries, order of work |
| Prediction interface: sequences in, predictions out | What to try next |

## Status

Early. Milestones 1 to 3 of 4 are in place: the project contract, the scoring rubric, the
hill-climbing loop with its budget and experiment records, the dashboard, the prediction page,
and all three skills. What is left is running the whole flow on a dataset from a different
problem, to find where the design is still shaped around the first one.

See [INSTALL.md](INSTALL.md) to set it up for Claude Code or Codex.

There is no sandbox. The agent runs with your permissions. Scores are protected against
accidents (test labels live outside the project folder), not against an agent that
deliberately goes looking for them.

## Install

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[app,examples,dev]"
```

## Reference project: amylase

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

`predict.py` reads a CSV with a `sequence` column (or a FASTA file) and writes a CSV with one
`prediction` per input row, in the same order. It is the only agent-written code with a fixed
shape, because both the rubric and the prediction page call it.

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

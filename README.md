# The-ML-Guy

Describe a modelling problem — something to predict, or something to generate — to a coding agent (Claude Code or Codex) and get back a trained
model, a report, and a web page for using it. No ML background needed.

The design rule: **be exact about the problem and the scoring, and leave the modelling open.**

| Strict: fixed when you approve the brief | Free: the agent's call |
| --- | --- |
| Task: input, output, prediction or generation | Model family and architecture |
| Data: train, validation and test sets | How the data is read, features and encodings |
| The evaluator (`task/evaluate.py`) and its direction | Hyperparameters and training procedure |
| Budget | Code structure, libraries, order of work |
| The command that runs the model: `infer.py` | What to try next |

## Philosophy

The-ML-Guy is a thin layer on top of a coding agent, not a framework that runs one.

- **Minimal, but load-bearing.** It supplies only what an agent cannot be trusted or relied on
  to do for itself: an honest score, a count of submissions that actually stops, state that
  survives the session, and a page other people can use. Everything else is left alone —
  including the compute, which the framework never sees and so cannot cap.
- **Maximum room for the agent.** The task, the data, the metric and the budget are fixed. The
  modelling is not: any architecture, any library, any code structure. Autoresearch is only as
  good as the freedom it has, so the tool stays out of it.
- **Skills before framework code.** For each new feature, first ask whether a short agent skill
  can deliver the user's goal. Add framework code only when reliable enforcement, shared
  interfaces, durable state, or reproducibility requires it. Skills guide judgment; code
  protects invariants.
- **Agent judgment for user-specific choices.** The framework reports the best validation
  score and keeps evaluation honest. The agent reviews the evidence and the user's priorities
  to choose the final model: highest score by default, or a comparably strong but simpler or
  more practical submission with an explicit recorded reason. No universal complexity
  penalty or hard-coded trade-off is imposed.
- **Domain- and format-agnostic.** The core knows nothing about proteins, cells, patients or
  images, does not depend on how a task stores its inputs, and knows nothing about what makes
  an output good beyond what the task tells it. Domain knowledge lives in the projects.
- **Agent-agnostic.** The contract is a command line and files on disk, so Claude Code, Codex
  or anything else can drive it, and a fresh session can pick up where the last one stopped.
- **No safety theatre.** Coding agents already carry their own controls. The-ML-Guy adds no
  sandbox and no permission layer; it protects scores from accidents, and says so plainly.

## Status

Early but complete end to end. All four milestones are in place: the project contract, the
scoring rubric, the hill-climbing loop with its budget and experiment records, the four
skills, and two worked examples from different problem shapes
(a variant landscape with a continuous target, and binary classification over unrelated
proteins of varying length).

See [INSTALL.md](INSTALL.md) to set it up for Claude Code or Codex.

There is no sandbox. The agent runs with your permissions. Scores are protected against
accidents (test labels live outside the project folder), not against an agent that
deliberately goes looking for them.

## Install

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .                        # add ".[examples,dev]" for the examples and tests
```

## What keeps the score honest

The rubric is the one thing the framework owns outright, so it is the one thing it is strict
about.

- **The task is frozen.** `task/` is hashed when the brief is approved; a changed task is
  refused, and cannot be re-frozen once anything has been scored against it.
- **The test set is scored once,** by `mlguy finish`, on the submission that did best on
  validation. No other command touches it.
- **The split is checked before it is frozen:** identical inputs in two splits, sets too small to
  measure anything, a validation set that looks unlike the test set.
- **One run, one score:** every submission is scored by the same frozen evaluation on the same
  validation set. When scores are about the same, the simpler model is preferred, and choosing
  it over the top score is recorded with a reason.
- **A failure is an error, never a zero,** and still uses up budget.

## Scoring

The framework has no built-in metrics. What counts as good is specific to each problem, and a
coding agent can write a correct scorer for it better than any fixed list can anticipate. So
every task carries its own evaluator, a script in `task/` that the agent writes, a fresh agent
audits, and the user approves; it is then frozen with the rest of the task.

```toml
[metric]
primary = "recall_at_top20pct"     # a name evaluate.py reports
direction = "maximize"
```

It is called as `python task/evaluate.py --outputs FILE --reference LABELS --inputs PATH
--split validation|final --result FILE` and writes `{"metrics": {...}}`. `--inputs` is what the
model was given for that split, so the score can depend on the example itself. The framework
decides which labels it is given and when: validation labels on every submission, test labels once, at the
end. `mlguy score --outputs FILE` runs it on any file, to check that it tells good outputs from
bad before the task is frozen.

**The audit.** Whoever builds the split and writes the evaluator is about to hill-climb on them,
so before they are frozen, a fresh agent with no stake in the result tries to break them:
degenerate outputs that score well, malformed ones that are not refused, leakage between or
inside the splits, a test set that does not match the intended use. Its report,
`task/audit.md`, goes to the user with the brief. `mlguy init` also refuses until the brief
records the user's approval.

**Generation is experimental.** A generative model has no answer key, so it always needs an
evaluator, and the same `infer.py` is asked for `--n N --seed S` samples per condition. The
mechanics are in place and tested on a toy problem; they have not yet been proven on a real
one. Prediction is what the framework is being hardened for first.

## Example: amylase

One-hot ridge regression on the [FLIP2](https://flip.protein.properties/) alpha-amylase
one-to-many split.

```bash
cd examples/amylase
python make_task.py                 # download the split, write task/data
mlguy init                          # check the task, hide the test labels, freeze task/
python src/train.py                 # this example's own baseline, into model/
mlguy validate                      # check prediction and scoring work; records experiment 0

# the loop: change the modelling however you like, then
python src/train.py --approach mutation-ridge
mlguy submit --note "ridge on substitution indicators instead of full-sequence one-hot"
mlguy status                        # what has been tried, what is best, what budget is left

mlguy finish                        # scores the best submission on test once, writes REPORT.md
python app.py                       # this project's own page
```

`mlguy submit` takes a snapshot of the code and the model and scores *that*, so what earned the
score cannot drift away from it afterwards — and so a model that only works in the working
folder is caught now rather than at the final check. It refuses once the submissions are spent.
The test set is scored exactly once,
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
├── task/            STRICT: brief.md (with the user's approval), task.toml, evaluate.py, audit.md, data
├── infer.py         FIXED INTERFACE: --model DIR --input PATH --output FILE
│                    (generation, experimental, adds --n N --seed S and writes n rows per condition)
├── src/             FREE: any code, any structure
├── model/           the model as it stands now
├── experiments/     written only by `mlguy submit`: log.jsonl + a snapshot per submission
├── final/           the scored submission, its test outputs and the test labels (by `mlguy finish`)
├── REPORT.md        written by `mlguy finish`
└── app.py           FREE: the page, built by the agent for this problem
```

Until `mlguy finish`, the test labels live in `~/.mlguy/projects/<name>/` (or under
`MLGUY_HOME`), outside the project folder, along with a hash of every file in `task/` taken when
the brief was approved. A project folder is a git repository; `submit` and `finish` commit to it —
but not `task/data`, which is usually too large and is fingerprinted rather than committed, so a
clone needs the data back from wherever the brief says it came from.

`infer.py` is the only agent-written code with a fixed shape, because both the rubric and the
project's page call it. `task/` is frozen and `src/` is free, so everything that decides what a
good model means is settled before any model is trained, and nothing about how to build one is.

**The framework does not depend on the form of a task's inputs.** They can be a table, an expression matrix, a
folder of images or structures; only the project's code has to open them. The framework knows two
things about an example, its id and its label, and matches outputs to labels by id.

```toml
[data]
train             = "data/train.h5ad"            # anything
validation_inputs = "data/validation.h5ad"       # anything
validation_labels = "data/validation_labels.csv" # id, target (generation: the conditions' ids alone)
test_inputs       = "data/test.h5ad"
test_labels       = "data/test_labels.csv"       # moved out of the project by `mlguy init`
```

There is no built-in page and no domain code in the core. The agent writes `app.py` for the
problem in front of it: the amylase example accepts FASTA and names each variant by its
mutations; a clinical model would look nothing like it.

## Gallery: what to say, and what came back

[`gallery/`](gallery/) holds five problems that have been built end to end — a clinical table,
single-cell annotation, pathology tiles, protein variant ranking, and a generative protein design
task — kept as **the request and the conversation**, not as code. Each one has the scientist's own
words, the questions the agent asked before it would freeze anything, and what came out with what
not to believe about it.

It is there because the modelling is the agent's job and your half of the contract is describing
the problem well enough to build a rubric from. That is the part with no reference material
anywhere else. `gallery/*/QUESTIONS.md` is the most useful file in each folder: the gap between
the request and what still had to be asked is the measure of how complete the request was.

## For agents

The `skills/` folder holds what a coding agent loads. Each skill states the goal, the rules and
the commands, and leaves the judgment to the agent.

| Skill | What it covers |
| --- | --- |
| `build-project` | Learn the machine, look at the data, ask only what is missing, write the split and the evaluator, get them audited and approved, freeze, validate |
| `audit` | Run by a fresh agent: try to break the split and the evaluator before they are frozen |
| `research` | The loop: the goal, the rubric, the budget, the commands. The approach is the agent's own |
| `deploy` | Score the test set once, write the report, build a page for the user's problem, record what was learned |

Starting a project by hand:

```bash
mlguy new my-project --name my-project   # skeleton: task.toml, brief.md, a stub infer.py
```

## Licence

MIT.

# How The-ML-Guy works

For contributors, and anyone who wants to know what happens underneath. Using it needs
none of this: see the [README](README.md).

The design rule: **be exact about the problem and the scoring, and leave the modelling open.**

| Strict: fixed when you approve the brief | Free: the agent's call |
| --- | --- |
| Task: what the model is given and what it returns | Model family and architecture |
| Data: train, validation and test sets | How the data is read, features and encodings |
| The evaluator (`task/evaluate.py`) and its direction | Hyperparameters and training procedure |
| Budget | Code structure, libraries, order of work |
| The command that runs the model: `infer.py` | What to try next |

## Philosophy

The-ML-Guy is a thin layer on top of a coding agent, not a framework that runs one.

Three parties, and what each owns: **you** decide what good means. **The agent** proposes how to
measure it, and a second agent tries to break that proposal. **The framework** makes what you
agreed impossible to change, and keeps the record. After that the agent owns all of the
modelling and all of the compute.

So the framework's code only ever does two things: it **refuses** (a changed task, a second
look at the test set, a submission once the search's time is up) and it **records** (the score, the code
that earned it, what it cost to run). Anything that would be advice lives in a skill instead,
where the agent can weigh it against the problem in front of it.

- **Minimal, but load-bearing.** It supplies only what an agent cannot be trusted or relied on
  to do for itself: an honest score, a time limit that actually stops the search, state that
  survives the session, and a page other people can use. Everything else is left alone —
  including the compute, which the framework never sees and so cannot cap.
- **Maximum room for the agent.** The task, the data, the metric and the budget are fixed. The
  modelling is not: any architecture, any library, any code structure. Autoresearch is only as
  good as the freedom it has, so the tool stays out of it.
- **Domain- and format-agnostic.** The core knows nothing about proteins, cells, patients or
  images, does not depend on how a task stores its inputs, and knows nothing about what makes
  an output good beyond what the task tells it. Domain knowledge lives in the projects.
- **Agent-agnostic.** The contract is a command line and files on disk, so Claude Code, Codex
  or anything else can drive it, and a fresh session can pick up where the last one stopped.
- **No safety theatre.** Coding agents already carry their own controls. The-ML-Guy adds no
  sandbox and no permission layer; it protects scores from accidents, and says so plainly.

## What keeps the score honest

Once the rubric is agreed, the framework holds it fixed. That is the one thing it is strict
about.

- **The task is frozen.** `task/` is hashed when the brief is approved; a changed task is
  refused, and cannot be re-frozen once anything has been scored against it.
- **The test set is scored once,** by `mlguy finish`, on the submission that did best on
  validation. No other command touches it.
- **The split and the evaluator are audited before they are frozen,** by an agent that did not
  write them. `mlguy init` refuses without the audit; skipping it takes a stated reason, which
  the report repeats.
- **One run, one score:** every submission is scored by the same frozen evaluation on the same
  validation set. When scores are about the same, the simpler model is preferred, and choosing
  it over the top score is recorded with a reason.
- **A failure is an error, never a zero,** and still uses up budget.
- **The budget that is enforced is time,** counted from the freeze. Submissions are counted and
  reported, not capped: the validation reference is in the project, so a cap would not stop the
  search looking, only stop it recording what it saw. A user who wants a cap can still set one.
  GPU-hours and money are written in the task and kept by the agent: the framework never sees
  the compute.

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

It is called as `python task/evaluate.py --outputs PATH [--reference PATH] [--inputs PATH]
--split validation|final --result FILE` and writes `{"metrics": {...}}`. The framework decides
which split it is given and when: validation on every submission, the test split once, at the
end. `mlguy score --outputs PATH` runs it on any outputs, to check that it tells good from bad
before the task is frozen. Defining it with the user is the `evaluation` skill.

**The audit.** Whoever builds the split and writes the evaluator is about to hill-climb on them,
so before they are frozen, a fresh agent with no stake in the result tries to break them:
degenerate outputs that score well, malformed ones that are not refused, leakage between or
inside the splits, a test set that does not match the intended use. Its report,
`task/audit.md`, goes to the user with the brief. The framework itself holds no opinion about a
split, so `mlguy init` refuses without the audit, and without the user's approval recorded in
the brief.

**One contract for every task.** The framework has no kinds of task: no prediction mode, no
generation mode, no metrics, no formats. A split is up to two things — inputs the model is given,
and a reference only the evaluator sees — and either can be anything. A task that samples fixes
its sample count and seed in `[infer] args`, and its evaluator enforces them; a task that ranks,
segments, forecasts or designs does whatever it needs the same way.

## Project layout

```
project/
├── task/            STRICT: brief.md (with the user's approval), task.toml, evaluate.py, audit.md, data
├── infer.py         FIXED INTERFACE: --model DIR [--input PATH] --output PATH [+ the task's [infer] args]
├── src/             FREE: any code, any structure
├── model/           the model as it stands now
├── experiments/     written only by `mlguy submit`: log.jsonl + a snapshot per submission
├── final/           the scored submission, its test outputs and the test reference (by `mlguy finish`)
├── REPORT.md        written by `mlguy finish`
└── app.py           FREE: the page, built by the agent for this problem
```

Until `mlguy finish`, the test split lives in `~/.mlguy/projects/<name>/` (or under
`MLGUY_HOME`), outside the project folder, along with a hash of every file in `task/` taken when
the brief was approved. A project folder is a git repository; `submit` and `finish` commit to it —
but not `task/data`, which is usually too large and is fingerprinted rather than committed, so a
clone needs the data back from wherever the brief says it came from.

`infer.py` is the only agent-written code with a fixed shape, because both the rubric and the
project's page call it. `task/` is frozen and `src/` is free, so everything that decides what a
good model means is settled before any model is trained, and nothing about how to build one is.

**The framework does not read a task's data.** Inputs and references can be a table, an
expression matrix, a folder of images or structures; only the project's code and its evaluator
open them.

```toml
[data]
train                = "data/train.h5ad"           # anything; never read by mlguy
validation_inputs    = "data/validation.h5ad"      # what infer.py is given
validation_reference = "data/validation_ref.csv"   # what only evaluate.py sees
test_inputs          = "data/test.h5ad"            # both test parts are moved out
test_reference       = "data/test_ref.csv"         #   of the project by `mlguy init`

[infer]
args = { n = 20, seed = 0 }                        # optional: passed to infer.py as --n 20 --seed 0
```

There is no built-in page and no domain code in the core. The agent writes `app.py` for the
problem in front of it: a protein-variant page might accept FASTA and name each variant by its
mutations; a clinical model would look nothing like it.

## Skills

The `skills/` folder holds what a coding agent loads. Each skill states the goal, the rules and
the commands, and leaves the judgment to the agent.

| Skill | What it covers |
| --- | --- |
| `build-project` | Learn the machine, look at the data, ask only what is missing, write the split and the evaluator, get them audited and approved, freeze, validate |
| `audit` | Run by a fresh agent: try to break the split and the evaluator before they are frozen |
| `research` | The loop: the goal, the rubric, the budget, the commands. The approach is the agent's own |
| `deploy` | Score the test set once, write the report, build a page for the user's problem, record what was learned |
| `evaluation` | Turn what the user means by success into a metric on a split, try to break it, and write `task/evaluate.py` |
| `environment` | Learn the user's machine — compute, scheduler, storage, where code and environments go, how to build one — prove it with a small job, and write it down once per machine |
| `research-scout` | Look up what is already known about the task — evaluation pitfalls, leaks, realistic scores, methods — and write it to `RESEARCH_CONTEXT.md` with sources |
| `writing` | Used by the others for anything the user reads: what they need first, in their words, with the depth kept and the padding cut |

Starting a project by hand:

```bash
mlguy new my-project --name my-project   # skeleton: task.toml and a stub infer.py
```

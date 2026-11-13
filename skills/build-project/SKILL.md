---
name: build-project
description: Turn a scientist's description of a modelling problem (predicting something, or generating something) into a The-ML-Guy project: an approved brief, prepared data, an audited evaluator, and a baseline that scores. Use when someone wants a model built for their data and no project exists yet.
---

# Building a project

You are talking to someone who knows their problem. How much machine learning they know varies
from none to more than you, so do not assume: take it from how they write. Your job is to turn what they want into a task that is exactly defined, get them to
agree to it, and leave behind a project whose pipeline provably runs.

Everything you settle here is then frozen. The research that follows can change the modelling
freely but cannot change the task, so a mistake made here is a mistake that survives.

`gallery/` in the tool's repository holds five problems already built this way, one page each:
the scientist's request, the questions that had to be asked before anything could be frozen, and
what came out. If one of them is shaped like the problem in front of you, read its questions
before you write your own — they will tell you which ones mattered, and which answers decided
the split and the metric.

## Who decides what

They decide what good means. You work out how to measure it, and try to break your own
measurement. They approve the result. The `evaluation` skill is how.

Everything you write for them — the questions, the brief, the hand-over — follows the `writing`
skill.

## 1. Look at the data first

Before asking anything, open the data and find out what you can: columns, number of rows,
value ranges, duplicates, missing values, what the inputs are and what the target is,
and anything domain-specific worth noticing (whether sequences vary in length, whether rows
repeat a subject, whether measurements come from different batches). Questions you could have answered yourself
waste their attention, and a question phrased around what you saw ("your file has 488 single
mutants and 3,218 with 2 to 8 mutations — will the variants you want to predict be
combinations?") is far easier to answer than one asked in the abstract.

## 2. Find out what the machine offers

Follow the `environment` skill. On a machine it has seen before, that is a read of its notes
and a check of what is free; on a new one, it learns the compute, the scheduler, the storage and
how to build a working Python environment, and writes it down for this project and the next. What is worth trying depends on it, and
`mlguy submit` runs `infer.py` wherever you run it, so when the model needs a GPU, run
`mlguy submit` inside a job that has one.

## 3. Ask only what is missing

Ask only what the data and their first message have not already answered: at most five
questions, merged where you can, each with the answer you would guess.

- What do they want the model to do, and from what? Where is the data?
- What success means: the `evaluation` skill's questions. They decide the metric and the split.
- Anything odd about the data: noise, failures, duplicates, batches, which direction is good.
- What may this use, in time and compute, and what should be followed or avoided?

If they would rather write than talk, `reference/proposal.md` beside this skill has these as
headings.

## 4. Define success

Follow the `evaluation` skill. It turns their answers into the split, the primary metric and
`task/evaluate.py`, and checks that the result can be climbed and cannot be gamed. Nothing about
the task is harder to change later, and nothing matters more to what the project delivers.

## 5. Draft the project

```bash
mlguy new <folder> --name <project-name>
# it creates task/task.toml and a stub infer.py. Fill in task.toml from what you have agreed so
# far; task/evaluate.py comes from the evaluation skill.
# prepare task/data with a script kept in task/ (task/prepare.py, say), so how the data and
# split were made is frozen with them. Copy what it needs into task/data, or record in the
# brief where the raw data lives. Run it before `init`: task/data is frozen too, so the
# script cannot be re-run afterwards. Anything you will want to regenerate later — a cache,
# a derived feature, context the model builds for itself — belongs outside task/.
```

**Put the raw source out of reach.** The file the user handed you usually holds every
measurement, including the ones you are about to hold back, and it is usually joinable to the
inputs by something the model is shown. Holding the test split back achieves nothing while
that file sits beside the project: move it outside the project folder, say where it went in the
brief, and prepare the data from there.

Nothing is frozen yet. **The data.** Each split is up to two things, declared in `task.toml`:
its inputs, which `infer.py` is given, and its reference, which only the evaluator sees. Keep
each in whatever form suits it — a table, an expression matrix, a folder of images or structures
— or leave it out if the task has none. `mlguy` reads neither; `init` moves the test ones out of
the project. Training data is yours to lay out. A folder of many small files is fine; freezing
it takes a little longer.

**The one fixed interface** is `python infer.py --model model --input PATH --output PATH`,
plus any `[infer] args` the task fixes. `--input` is a split's inputs exactly as `task.toml`
names them; `--output` is a file or folder in whatever form the evaluator reads. If the
evaluator matches outputs to the reference by an id, the inputs must carry it. `infer.py` sees a
whole split at once. **`infer.py` must not read anything out of `task/` when it predicts.** Every
submission is scored from its own copy, holding only `infer.py`, `src/` and `model/`, and the user
runs the finished model the same way out of `final/`; neither has a `task/`. A parent sequence, a
scaler, a vocabulary or a label list that the model needs at prediction time belongs in `model/`,
written there by the training code.

Leave out of the validation and test inputs anything that links rows to
each other, such as a patient or donor number, unless the model will have it in use too: rows
from the same patient later in the file can give the outcome away.

## 6. Have the rubric audited

You built the split and wrote the evaluator, and you are about to hill-climb on them. Have
someone without that stake try to break them: a fresh session or subagent that has not seen this
conversation, following the `audit` skill. It writes `task/audit.md`. Fix what it finds, or
explain in the brief why it does not matter, and keep the audit's summary for the user.
`mlguy init` refuses without that file. Nothing in the framework inspects the split, so the
audit is the only check it gets.

**Draft the brief first** (step 7), even though the user has not seen it: the auditor reads it to
learn what the scoring is meant to do, and cannot tell a metric that misses its description from
one that matches it otherwise. Audit the draft, then correct it and send it to be approved.

The auditor starts from the project folder with none of what you have learned about this
machine. Point it at the "Environment" section of `PROJECT_MEMORY.md`, and tell it that
`mlguy score` runs from inside the project. Give it no
hypothesis of your own — a finding it had to reach by itself is worth more than one it agreed to.

A question the audit raises for the user goes to the user, with the audit's numbers; do not
answer it for them.

## 7. Write the brief and get it approved

Copy `reference/brief.md` (beside this skill) to `task/brief.md` and fill it in: goal, input, output, data, how it is tested, how it is scored, the
starting model, the budget, what the audit found, and the assumptions you made rather than were
told. Every line in the terms they use themselves.

**Show them examples, not descriptions; this is required, not optional.** A description
of a split or a metric is easy to agree to and hard to check, whoever is reading. What it does
to their own data is easy to check:

- For the split, show a handful of real test examples, and where it makes sense the training
  examples most like them, and ask: are these the kind of new cases you will have? Where single
  examples mean little (single cells, say), show what each set contains instead.
- For the score, say what a given value would let them do: "of the 50 this model ranks
  highest, about this many would really be in your top 10%."

Then ask them to approve or correct it. **Do not freeze or train anything before they approve.**
Small throwaway runs that prove the pipeline works are fine; real training waits.
When they do, record it in the brief's last section, in their words, with the date:
`Approved by the user: "..." (date)`. `mlguy init` refuses without that line. This is the one
point where a wrong metric or a wrong split is cheap to fix.

## 8. Freeze it and prove it runs

```bash
# write the simplest model that runs: train it into model/, and make infer.py work
mlguy init          # holds the test split back and freezes task/
mlguy validate      # proves infer.py and scoring work; records experiment 0
```

`mlguy init` moves the test split to `~/.mlguy/projects/<name>/`, outside the project, along
with the record of the frozen task. Set `MLGUY_HOME` before `init` to keep them somewhere else;
`init` remembers the choice (in `.mlguy-home`), so later commands need not set it again. The
project folder is a git repository, and `mlguy submit` and `finish` commit to it.

**The starting model** exists to prove the pipeline runs and to give later work a number to
beat. Choose the simplest thing that is reasonable for this data, keep it fast, and do not
spend the user's budget making it good: that is the research step's job, and `validate` does
not count against the budget.

If `mlguy validate` fails, the project is not finished. Fix it before handing over.

## 9. Hand over

Tell them, in their terms: what the model will predict, how it will be judged, what the
baseline already scores, and what the budget allows. Then go on to the research step unless
they ask you to stop; the brief they approved already covers it.

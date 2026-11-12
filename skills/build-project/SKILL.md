---
name: build-project
description: Turn a scientist's description of a modelling problem (predicting something, or generating something) into a The-ML-Guy project: an approved brief, prepared data, an audited evaluator, and a baseline that scores. Use when someone wants a model built for their data and no project exists yet.
---

# Building a project

You are talking to a scientist who knows their problem and may know nothing about machine
learning. Your job is to turn what they want into a task that is exactly defined, get them to
agree to it, and leave behind a project whose pipeline provably runs.

Everything you settle here is then frozen. The research that follows can change the modelling
freely but cannot change the task, so a mistake made here is a mistake that survives.

## The rule about language

They speak about their science. You translate to machine learning. They approve the
translation. Never ask them to choose a metric, a split, a model or a loss by name. Ask what
they will do with the predictions and what the new inputs will look like, and derive the rest
yourself.

## 1. Look at the data first

Before asking anything, open the data and find out what you can: columns, number of rows,
value ranges, duplicates, missing values, what the inputs are and what the target is,
and anything domain-specific worth noticing (whether sequences vary in length, whether rows
repeat a subject, whether measurements come from different batches). Questions you could have answered yourself
waste their attention, and a question phrased around what you saw ("your file has 488 single
mutants and 3,218 with 2 to 8 mutations — will the variants you want to predict be
combinations?") is far easier to answer than one asked in the abstract.

## 2. Find out what the machine offers

Before planning any modelling, learn the compute you are working with, because it decides what
is worth trying. This is yours to discover, not the user's to explain: CPUs and memory, GPUs and
their type, whether there is a job scheduler (Slurm: `sinfo`, `sacctmgr show assoc user=$USER`,
`squeue`), which partitions and accounts you may use and their limits, and where large files and
environments should live. Check your own memory and the machine's notes (`CLAUDE.md`,
`AGENTS.md`) first; someone may have written it down already. Ask the user only what cannot be
discovered: which account to charge, what they are willing to spend.

Write what you found in `PROJECT_MEMORY.md` under "Compute", and into your own memory if you
keep one, so the next session does not have to rediscover it. The framework does not manage
compute: `mlguy submit` runs `infer.py` wherever you run it, so when the model needs a GPU,
run `mlguy submit` inside a job that has one.

## 3. Ask only what is missing

Below is what only they can tell you. It is not a script: by now the data and their first
message have answered some of it, and you should not spend their attention on those. Ask at
most five questions in all, merging where you can, each with the answer you would guess.

1. What do you want to predict, and from what?
2. Where is the data?
3. **What will you do with the predictions?** This fixes the metric. Ranking candidates to
   test leads to a rank metric; needing the number itself leads to an error metric.
4. **What will new inputs look like, compared with the data you have?** This fixes the split.
   It is the question they are least likely to volunteer and the one that most often decides
   whether the model is useful.
5. Anything odd about the measurements? Noise, failed assays, duplicates, batches, and which
   direction is good.
6. How much time and compute may this use?
7. Anything you want followed or avoided, and what would be good enough?

If they would rather write than talk, `src/mlguy/templates/proposal.md` in the tool's repository has
these as headings.

## 4. Choose the split with care

Build the split so that scoring on it resembles the use they described when they said what new
inputs would look like. For example, a random
split is almost always wrong when rows are related to each other — similar sequences, repeat
measurements on the same subject, cells from one donor: near-duplicates land on both sides
and every number comes out flattering.

Once the test set is drawn, do not score anything on it, not even a simple rule for the
brief. Every number in the brief comes from training or validation data; the test set is
opened once, by `mlguy finish`.

Make the validation set resemble the test set. Hill-climbing happens on validation, so if
validation asks an easier or simply different question, the search will optimise for the wrong
thing and the final score will disappoint. When the split makes that unavoidable — because a
published split is being reproduced, or the data only allows one sensible division — say so in
the brief, and set `validation_matches_test = false` with a `validation_note` in `task.toml`.
The warning then follows through to the report. A caveat that is not a mismatch (validation and
test share patients, say) also goes in `validation_note`; it reaches the report either way.

`validation_note` is frozen with the rest of `task.toml` and quoted **word for word into the
report the user reads**, so write it as prose for them and get it right the first time: a mistake
in it cannot be corrected afterwards, only apologised for underneath.

## 5. Write the evaluator

Every task carries its own scoring, `task/evaluate.py`, frozen with the task. The framework has
no built-in metrics: what counts as good is specific to the problem, and you are better placed
than any fixed list to write it.

```
python evaluate.py --outputs FILE --reference LABELS --inputs PATH \
                   --split validation|final --result FILE
```

It reads the model's outputs and the labels file of that split, matched by `id`, and writes
`{"metrics": {"name": number, ...}}`. `--inputs` is the same thing the model was given, so a
score may depend on what each example is: the backbone a design should match, a field to group
by. For the final check it points at the held-back inputs wherever they are kept, so use it
rather than a path of your own. Accept all five arguments even if you ignore one. The primary
metric must be a number; others may be `NaN` when undefined. Set `[metric] primary` and `direction` in `task.toml`. Training code may import
from it to tune against the same measure. Refuse outputs that cannot be scored (missing or
repeated ids, wrong columns) by exiting with an error; never score them as zero.

**Derive the metric from what they will do with the predictions,** then check it against the
split. A metric that moves with something other than model quality will mislead the search: when
the class balance differs between validation and test, plain accuracy measures the base rate as
much as the model, and a balanced measure is the safer choice. If you keep a base-rate-dependent
metric anyway (to stay comparable with published numbers, say), record why in the brief. Use a
well-tested library for standard metrics rather than writing them by hand.

**For generation, how good outputs are recognised is the user's knowledge, not yours.** Ask how
they would judge a batch of outputs by hand, and what they already trust to do it: a folding
model, a classifier, a simulation, a comparison with real data. (Generative tasks are
experimental; say so if a user asks for one.) A generative task has no labels: its
`validation_labels` and `test_labels` files list the conditions to sample for, so the framework
knows how many samples to expect for each, and the conditions themselves stay in the inputs.

**If the scoring needs a different environment from the model** — a folding model that will not
coexist with your training stack, a tool in another language — `evaluate.py` is still the entry
point, called with the interpreter `mlguy` runs under. Have it start the other environment
itself. Set `[evaluator] timeout_minutes` to fit: scoring runs on every submission, and the
budget is in wall-clock as much as in submissions.

The search will optimise exactly what this script rewards, so:

- **It must not be winnable by degenerate outputs.** A constant prediction, or one good sample
  repeated, must not score well. Fold validity, diversity and novelty into the primary number
  for generation, or refuse outputs that fail them.
- **Prove it can tell good from bad.** Run `mlguy score --outputs FILE` on something good (the
  validation labels themselves, where that makes sense) and on something broken (shuffled,
  constant, random). If the scores do not separate, it cannot guide a search.
- **Check the metric is not already saturated.** Train the simplest reasonable model and score it
  now, before the freeze, rather than at step 9. If that baseline already sits near the best
  possible value, the entire budget can only move the number by the little left over, and most of
  that will be noise: the project is finished before it starts. A ceiling found now is fixed by a
  harder metric or a harder split; found afterwards it cannot be fixed at all, because the task
  cannot change once anything is scored. In one trial a sixteen-second ridge regression scored
  0.94 of a possible 1.00, which nothing in the twelve submissions that followed could improve on.
- **Prove it can resolve a real difference, and not merely a broken one.** Good against broken is
  a floor check; the search needs the metric to separate two plausible models. If anything is
  already known to help on this problem — a published model, a standard baseline — score it
  against the obvious default and ask whether the gap is larger than the spread this many
  examples would give you anyway. A metric that cannot see the intervention the field already
  trusts will not see yours. How many examples each split holds is part of the same question: a
  headline that moves in steps of 1/32 cannot show a 2% improvement. Both are free to change now
  and impossible to change later.
- **It runs on every submission,** so its cost is part of the budget.

## 6. Draft the project

```bash
mlguy new <folder> --name <project-name>
# task/task.toml from what you have agreed so far; task/evaluate.py as above
# prepare task/data with a script kept in task/ (task/prepare.py, say), so how the data and
# split were made is frozen with them. Copy what it needs into task/data, or record in the
# brief where the raw data lives.
```

**Put the raw source out of reach.** The file the user handed you usually holds every
measurement, including the ones you are about to hold back, and it is usually joinable to the
inputs by something the model is shown. Hiding `test_labels.csv` achieves nothing while that
file sits beside the project: move it outside the project folder, say where it went in the
brief, and prepare the data from there. Every audit so far has caught this.

Nothing is frozen yet. **The data.** Keep the inputs in whatever form suits them: a table, an
expression matrix, a folder of images or structures. `mlguy` never reads them; only your code
does. What it needs is an `id` for every validation and test example, carried by the inputs, and
a small labels CSV for each of those two splits with `id` and `target`. Training data is yours
to lay out. A folder of many small files is fine; freezing it takes a little longer.

**The one fixed interface** is `python infer.py --model model --input PATH --output FILE`,
where `PATH` is a split's inputs exactly as `task.toml` names them, and the output is a CSV
with `id` and whatever your evaluator reads (by convention `prediction`). `infer.py` only ever
sees inputs, so the inputs must carry the `id`; the model should ignore it. It also sees a
whole split at once. **`infer.py` must not read anything out of `task/` when it predicts.** Every
submission is scored from its own copy, holding only `infer.py`, `src/` and `model/`, and the user
runs the finished model the same way out of `final/`; neither has a `task/`. A parent sequence, a
scaler, a vocabulary or a label list that the model needs at prediction time belongs in `model/`,
written there by the training code.

Leave out of the validation and test inputs anything that links rows to
each other, such as a patient or donor number, unless the model will have it in use too: rows
from the same patient later in the file can give the outcome away.

## 7. Have the rubric audited

You built the split and wrote the evaluator, and you are about to hill-climb on them. Have
someone without that stake try to break them: a fresh session or subagent that has not seen this
conversation, following the `audit` skill. It writes `task/audit.md`. Fix what it finds, or
explain in the brief why it does not matter, and keep the audit's summary for the user.

**Draft the brief first** (step 8), even though the user has not seen it: the auditor reads it to
learn what the scoring is meant to do, and cannot tell a metric that misses its description from
one that matches it otherwise. Audit the draft, then correct it and send it to be approved.

The auditor starts from the project folder, but it has none of what you have learned about this
machine, so tell it the rest: which interpreter or virtual environment to use, that `mlguy score`
runs from inside the project, where to put scratch files, and what compute it has. Give it no
hypothesis of your own — a finding it had to reach by itself is worth more than one it agreed to.

## 8. Write the brief and get it approved

Fill in `task/brief.md`: goal, input, output, data, how it is tested, how it is scored, the
starting model, the budget, what the audit found, and the assumptions you made rather than were
told. Every line in their language.

**Show them examples, not descriptions; this is required, not optional.** They cannot judge
"held out by similarity" or "Spearman correlation", but they can judge what those mean on their
own data:

- For the split, show a handful of real test examples, and where it makes sense the training
  examples most like them, and ask: are these the kind of new cases you will have? Where single
  examples mean little (single cells, say), show what each set contains instead.
- For the score, say what a given value would let them do: "of the 50 this model ranks
  highest, about this many would really be in your top 10%."

Then ask them to approve or correct it. **Do not freeze or train anything before they approve.**
When they do, record it in the brief's last section, in their words, with the date:
`Approved by the user: "..." (date)`. `mlguy init` refuses without that line. This is the one
point where a wrong metric or a wrong split is cheap to fix.

## 9. Freeze it and prove it runs

```bash
# write the simplest model that runs: train it into model/, and make infer.py work
mlguy init          # checks the task and the split, moves the test labels out, freezes task/
mlguy validate      # proves infer.py and scoring work; records experiment 0
```

`mlguy init` moves the test labels to `~/.mlguy/projects/<name>/`, outside the project, along
with the record of the frozen task. Set `MLGUY_HOME` before `init` to keep them somewhere else;
`init` remembers the choice (in `.mlguy-home`), so later commands need not set it again. The
project folder is a git repository, and `mlguy submit` and `finish` commit to it.

**The split is checked** before it is frozen: identical inputs in two splits, a validation or
test set too small to measure anything, a validation set whose labels look unlike the test
set's. Each `check:` line `mlguy init` prints is a way the final score could mislead. Fix it,
or explain it in the brief and to the user, before going on; once anything is scored, the task
can no longer be changed.

**The starting model** exists to prove the pipeline runs and to give later work a number to
beat. Choose the simplest thing that is reasonable for this data, keep it fast, and do not
spend the user's budget making it good: that is the research step's job, and `validate` does
not count against the budget.

If `mlguy validate` fails, the project is not finished. Fix it before handing over.

## 10. Hand over

Tell them, in their language: what the model will predict, how it will be judged, what the
baseline already scores, and what the budget allows. Then go on to the research step unless
they ask you to stop; the brief they approved already covers it.

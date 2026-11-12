---
name: build-project
description: Turn a scientist's description of a modelling problem (predicting something, or generating something) into a The-ML-Guy project: an approved brief, prepared data, and a baseline that scores. Use when someone wants a model built for their data and no project exists yet.
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

## 2. Ask only what is missing

These are the things only they can tell you. Skip any the data or their first message already
answers. Ask at most five questions in all, merging where you can, each with the answer you
would guess.

1. What do you want to predict, and from what?
2. Where is the data?
3. **What will you do with the predictions?** This fixes the metric. Ranking candidates to
   test leads to a rank metric; needing the number itself leads to an error metric.
4. **What will new inputs look like, compared with the data you have?** This fixes the split.
   It is the question they are least likely to volunteer and the one that most often decides
   whether the model is useful.
5. Anything odd about the measurements? Noise, failed assays, duplicates, batches, and which
   direction is good.
6. How much time and compute can this use?
7. Anything you want followed or avoided, and what would be good enough?

If they would rather write than talk, `src/mlguy/templates/proposal.md` in the tool's repository has
these as headings.

## 3. Choose the split with care

Build the split so that scoring on it resembles the use they described in question 4. For example, a random
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
The warning then follows through to the report.

## 4. Choose a metric that cannot be gamed by the split

Pick the metric from what they will do with the predictions, then check it against the split
you just chose. A metric that moves with something other than model quality will mislead the
search.

The common trap is a class balance that differs between validation and test: plain accuracy
then measures the base rate as much as the model, and the loop will prefer a model fitted to
the wrong one. When the balance differs, prefer `balanced_accuracy` or `mcc` over `accuracy`.
All the classification metrics take any number of classes, with labels as numbers or text,
except `roc_auc`, which needs exactly two.
If you keep a base-rate-dependent metric anyway — to stay comparable with published numbers,
say — record that reason in the brief, because the search will be ranked by it.

## 5. When no built-in metric fits, write the evaluator

The built-in metrics cover one prediction per example scored against a label. When the real
measure of success is something else, the task carries its own evaluator. (Generative tasks
always need one. Support for them is experimental; say so if a user asks for one.)

**How good outputs are recognised is the user's knowledge, not yours.** Ask how they would
judge a batch of outputs by hand, and what they already trust to do it: a folding model, a
classifier, a simulation, a comparison with real data. Turn that into `task/evaluate.py`,
describe in the brief what it rewards in plain words, and have them approve it. It is frozen
with the task.

```
python evaluate.py --outputs FILE --reference LABELS --split validation|final --result FILE
```

It reads the model's outputs and the labels file of that split, matched by `id`, and writes
`{"metrics": {"name": number, ...}}`. The primary metric must be a number; others may be
`NaN` when undefined. Set
`[evaluator] script`, and `[metric] primary` and `direction`, in `task.toml`. Training code may
import from it to tune against the same measure.

The search will optimise exactly what this script rewards, so:

- **It must not be winnable by degenerate outputs.** A score for quality alone is maximised by
  one good sample repeated. Fold validity, diversity and novelty against the training data
  into the primary number, or refuse outputs that fail them.
- **Prove it can tell good from bad before freezing it.** Run `mlguy score --outputs FILE` on
  real held-out data and on something broken (shuffled, random, copied from training). If the
  scores do not separate, it cannot guide a search.
- **It runs on every submission,** so its cost is part of the budget.

## 6. Write the brief and get it approved

Fill in `task/brief.md`: goal, input, output, data, how it is tested, how it is scored, the
starting model, the budget, and the assumptions you made rather than were told. Every line in
their language.

**Show them examples, not descriptions; this is required, not optional.** They cannot judge "held out by similarity" or
"Spearman correlation", but they can judge what those mean on their own data:

- For the split, show a handful of real test examples, and where it makes sense the training
  examples most like them, and ask: are these the kind of new cases you will have? Where single
  examples mean little (single cells, say), show what each set contains instead.
- For the score, say what a given value would let them do: "of the 50 this model ranks
  highest, about this many would really be in your top 10%."

Then ask them to approve or correct it. **Do not build anything before they approve.** Drafting
`task.toml` or the evaluator to show them is fine; nothing is frozen until `mlguy init`. This is
the one point where a wrong metric or a wrong split is cheap to fix.

## 7. Build it

```bash
mlguy new <folder> --name <project-name>
# write task/task.toml from the approved brief; prepare task/data with a script kept in
# task/ (task/prepare.py, say), so how the data and split were made is frozen with them.
# Copy what it needs into task/data, or record in the brief where the raw data lives.
# write the simplest model that runs: train it into model/, and make infer.py work
mlguy init          # checks the task and the split, moves the test labels out, freezes task/
mlguy validate      # proves infer.py and scoring work; records experiment 0
```

**The data.** Keep the inputs in whatever form suits them: a table, an expression matrix, a
folder of images or structures. `mlguy` never reads them; only your code does. What it needs is
an `id` for every validation and test example, carried by the inputs, and a small labels CSV
for each of those two splits with `id` and `target`. Training data is yours to lay out. A
folder of many small files is fine; freezing it takes a little longer.

**The one fixed interface** is `python infer.py --model model --input PATH --output FILE`,
where `PATH` is a split's inputs exactly as `task.toml` names them, and the output is a CSV
with `id` and `prediction` (plus `score`, the positive-class probability, for two-class tasks).
`infer.py` only ever sees inputs, so the inputs must carry the `id`; the model should ignore it.
It also sees a whole split at once. Leave out of the validation and test inputs anything that
links rows to each other, such as a patient or donor number, unless the model will have it in
use too: rows from the same patient later in the file can give the outcome away.

A caveat about the split that is not a validation/test mismatch (validation and test share
patients, say) goes in `[data] validation_note`; it is carried into the report either way.

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

## 8. Hand over

Tell them, in their language: what the model will predict, how it will be judged, what the
baseline already scores, and what the budget allows. Then go on to the research step unless
they ask you to stop; the brief they approved already covers it.

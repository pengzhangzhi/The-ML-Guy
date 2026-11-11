---
name: build-project
description: Turn a scientist's description of a prediction problem into a The-ML-Guy project: an approved brief, prepared data, and a baseline that scores. Use when someone wants a model built for their data and no project exists yet.
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
value ranges, duplicates, missing values, which columns are inputs and which is the target,
and anything domain-specific worth noticing (whether sequences vary in length, whether rows
repeat a subject, whether measurements come from different batches). Questions you could have answered yourself
waste their attention, and a question phrased around what you saw ("your file has 488 single
mutants and 3,218 with 2 to 8 mutations — will the variants you want to predict be
combinations?") is far easier to answer than one asked in the abstract.

## 2. Ask only what is missing

These are the things only they can tell you. Skip any the data or their first message already
answers, and ask at most five, each with the answer you would guess.

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

`proposal.md` in a new project has these as headings, if they would rather write than talk.

## 3. Choose the split with care

Build the split so that scoring on it resembles the use they described in question 4. For example, a random
split is almost always wrong when rows are related to each other — similar sequences, repeat
measurements on the same subject, cells from one donor: near-duplicates land on both sides
and every number comes out flattering. 

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
If you keep a base-rate-dependent metric anyway — to stay comparable with published numbers,
say — record that reason in the brief, because the search will be ranked by it.

## 5. Write the brief and get it approved

Fill in `task/brief.md`: goal, input, output, data, how it is tested, how it is scored, the
starting model, the budget, and the assumptions you made rather than were told. Every line in
their language. Then show it and ask them to approve or correct it.

**Do not build anything before they approve.** This is the one point where a wrong metric or a
wrong split is cheap to fix.

## 6. Build it

```bash
mlguy new <folder> --name <project-name>
# write task/task.toml from the approved brief; prepare task/data/{train,validation,test}.csv
mlguy init          # checks the task, hides the test labels, freezes task/
python src/train.py # the baseline
mlguy validate      # proves prediction and scoring work; records experiment 0
```

The baseline exists to prove the pipeline runs and to give later work a number to beat. Keep
it simple and fast. Do not spend the user's budget trying to make it good — that is the
research step's job, and `validate` does not count against the budget.

If `mlguy validate` fails, the project is not finished. Fix it before handing over.

## 7. Hand over

Tell them, in their language: what the model will predict, how it will be judged, what the
baseline already scores, and what the budget allows. Then start the research loop, or tell
them to.

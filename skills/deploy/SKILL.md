---
name: deploy
description: Finish a The-ML-Guy project: score the test set once, write the report and build a page for the user's problem, and record what was learned. Use when the budget is spent or the user wants to stop and use the model.
---

# Finishing a project

The work is only useful if the scientist can run it and knows what to trust. Two things stand
between the experiments and that: a report they can read, and a page they can use.

```bash
mlguy finish    # runs the final check on the best submission, once, and writes REPORT.md
mlguy serve     # runs the page you build: app.py
```

## The report

`mlguy finish` writes `REPORT.md` with the scores and the experiment table. Your job is to make
the top of it answer the question they actually asked, in the words of `task/brief.md`.

- **Lead with what the number means for them,** not with the number. A Spearman of 0.63 means
  little; "if you pick the 50 variants this model ranks highest, about N of them are in the
  true top 10%" is something they can act on. Compute that from the test predictions rather
  than asserting it.
- **Say what it cannot do.** The split defines what was shown to work. A model tested on
  single mutants has not been shown to work on combinations, and saying so is part of the job.
- **Carry the caveat forward.** If `validation_matches_test` is false, the report already
  carries a warning; make sure it is in words they will understand.
- **Be straight when nothing improved.** If no submission beat the baseline, say that plainly
  and say what you would try next. A report that oversells is worse than no report.

## The page

There is no built-in page. You build `app.py` for this user and this problem, because only
you know what they will actually do with the model: rank candidates, check one case at a
time, compare two conditions, draw samples. Re-read `task/brief.md` and design for that. The
layout, the inputs it accepts, the plots and the library are all your call.

Three things are not:

- **It runs the scored model.** `mlguy finish` puts the best submission in `final/`. Get outputs by
  calling `final/infer.py`, the code path the report's
  numbers were measured on, not a reimplementation.
- **It says what the model was tested on and what it was not,** in their words, where someone
  using it cold will see it.
- **`python app.py [--port N] [--share]` starts it,** so `mlguy serve` works.

Accept input in the form they already have it, and when a row cannot be used, say why in
their terms. Check it yourself before handing over: start it, feed it real data, and feed it
something broken.

## Project memory

Write `PROJECT_MEMORY.md` for whoever comes next: what worked, what did not and why you think
so, what you would try with more budget, and anything about the data worth knowing. Be
concrete. "Gradient boosting scored 0.27 against ridge's 0.69, probably because 1,375 sparse
binary features and 2,574 rows is the wrong shape for trees" is worth keeping. "Tried several
models" is not.

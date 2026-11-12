---
name: deploy
description: Finish a The-ML-Guy project: score the test set once, write the report and build a page for the user's problem, and record what was learned. Use when the research is done: the budget is spent, more would not help, or the user wants to stop.
---

# Finishing a project

The work is only useful if the scientist can run it and knows what to trust. Two things stand
between the experiments and that: a report they can read, and a page they can use.

```bash
mlguy finish       # runs the final check on the best submission, once, and writes REPORT.md
python app.py      # the page you build
```

## The report

`mlguy finish` writes `REPORT.md` with the scores and the experiment table, and puts the scored
submission in `final/`, together with its outputs on the test split (`test_outputs`) and the
test reference beside them; `finish` prints where the test inputs are kept. The test split is
spent once it is scored, so use all of it freely
from here on. Your job is to make the top of `REPORT.md` answer the question they actually
asked, in the words of `task/brief.md`; edit it as you see fit, since `finish` will not write it
again. The `writing` skill covers how; what follows is what the report must say.

- **Lead with what the number means for them,** not with the number. A Spearman of 0.63 means
  little; "if you pick the 50 variants this model ranks highest, about N of them are in the
  true top 10%" is something they can act on. Compute that from the test outputs rather
  than asserting it.
- **Say what it cannot do.** The split defines what was shown to work. A model tested on
  single mutants has not been shown to work on combinations, and saying so is part of the job.
- **Carry the caveats forward.** The report quotes `validation_note` and points at
  `task/audit.md`; say what both mean for this result, in words they will understand.
- **Say how far the search moved it.** The table lists the starting model as experiment 0; put
  the change from there to the result in a sentence.
- **Be straight when nothing improved.** If no submission beat the baseline, say that plainly
  and say what you would try next. A report that oversells is worse than no report.

## The page

There is no built-in page. You build `app.py` for this user and this problem, because only
you know what they will actually do with the model: rank candidates, check one case at a
time, compare two conditions, draw samples. Re-read `task/brief.md` and design for that. The
layout, the inputs it accepts, the plots and the library are all your call. Nothing for it is
installed with the tool: install whatever it needs.

Three things are not:

- **It runs the scored model.** Get outputs by calling `final/infer.py`, the code path the
  report's numbers were measured on, not a reimplementation. If the page needs more from the
  model than a prediction, have `infer.py` write it as extra columns.
- **It says what the model was tested on and what it was not,** in their words, where someone
  using it cold will see it.
- **`python app.py [--port N]` starts it,** so anyone can run it the same way.

**`final/` holds only `infer.py`, `src/` and `model/`.** That is what was scored, and it is what
the user will copy and run. Anything the model reaches for outside it — a weights cache, a folding
model, a shared data folder — has to be found by an absolute path or a named environment variable,
and the report has to say which one and what to set it to. Test it from `final/` itself, not from
the working folder: a path that resolves only where you built the model works for you and for
nobody else.

The text on the page follows the `writing` skill too. Accept input in the form they already have it, and when a row cannot be used, say why in
their terms. Check it yourself before handing over: start it, feed it real data, and feed it
something broken.

## Project memory

Write `PROJECT_MEMORY.md` for whoever comes next: what worked, what did not and why you think
so, what you would try with more budget, and anything about the data worth knowing. Be
concrete. "Gradient boosting scored 0.27 against ridge's 0.69, probably because 1,375 sparse
binary features and 2,574 rows is the wrong shape for trees" is worth keeping. "Tried several
models" is not.

When the report, the page and the memory are done, commit them to the project's git repository.

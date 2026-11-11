---
name: deploy
description: Finish a The-ML-Guy project: score the test set once, write the report and the prediction page in the user's language, and record what was learned. Use when the budget is spent or the user wants to stop and use the model.
---

# Finishing a project

The work is only useful if the scientist can run it and knows what to trust. Two things stand
between the experiments and that: a report they can read, and a page they can use.

```bash
mlguy finish    # scores the best submission on the test set once, writes REPORT.md
mlguy serve     # the prediction page
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

## The prediction page

`mlguy serve` runs the generic page: upload a CSV of the task's input columns, get ranked
predictions, a plot and a download. It reads its wording from `[page]` in `task.toml`, so adjust that rather than
rewriting the page:

- `title`: what it predicts, named as they would name it.
- `row_name`: what one row is called, so the page says "42 variants" or "42 patients" rather
  than "42 rows".
- `description`: one or two sentences a colleague could read cold.
- `tested`: how it was tested, so nobody applies it to the wrong thing.

Check it yourself before handing it over: start it, upload a few rows of real data, and see
that the output is sensible and the rejected-row messages make sense.

## Project memory

Write `PROJECT_MEMORY.md` for whoever comes next: what worked, what did not and why you think
so, what you would try with more budget, and anything about the data worth knowing. Be
concrete. "Gradient boosting scored 0.27 against ridge's 0.69, probably because 1,375 sparse
binary features and 2,574 rows is the wrong shape for trees" is worth keeping. "Tried several
models" is not.

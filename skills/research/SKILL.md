---
name: research
description: Improve a model's score on a The-ML-Guy project, within its budget. Use when a project has been initialised with `mlguy init` and validated, and the user wants the model made better.
---

# Improving the model

Your job is to raise the score on the project's validation set, within its budget. How you do
that is entirely your call.

## What is fixed

Read `task/brief.md` and `task/task.toml` first. The task, the data, the split, the metric and
the budget are settled and must not be edited — `mlguy` checks `task/` on every submission and
refuses if it changed. One more thing is fixed:

```
python predict.py --model model --input rows.csv --output predictions.csv
```

It reads a CSV holding the task's input columns and writes a CSV with one `prediction` per
input row, in the same order (plus a `score` column for classification). The rubric and the prediction page both call it, so it must
keep working.

## What is yours

Everything else: the model, the features, the architecture, the loss, hyperparameters, how you
train, which libraries you use, how you organise `src/`. Anything that fits the budget is fair
game. You are not limited to variations on the baseline, and the baseline is a starting point,
not a template.

## The loop

```bash
mlguy status                       # what has been tried, what is best, what budget is left
# ... change whatever you want, retrain into model/ ...
mlguy submit --note "one line on what you changed and why"
mlguy finish                       # when the budget is spent
```

`mlguy submit` scores `model/` on validation, records the experiment with a snapshot of the
code and model, and tells you the score and the remaining budget. It refuses once the budget is
spent. A crash or a malformed predictions file is recorded as an error, not as a score of zero,
but it still uses up a submission, so test your prediction command before you submit.

Never score the test set. `mlguy finish` does that once, at the end, on whichever submission
scored best on validation. Looking at it earlier makes the final number meaningless.

## Working well

- Spend your first move understanding the data: the split, the shape of the target, what makes
  validation different from training. The brief says what the user actually wants to do with
  the predictions.
- Say what you expect before you run something, then compare. A submission that falls flat but
  was predicted to is worth more than a lucky one.
- The `--note` is the only record a human reads later. "ESM-2 embeddings instead of one-hot,
  expecting better transfer to unseen positions" beats "try new model".
- Watch for overfitting to validation, especially late in a budget, and prefer a change that
  should generalise over one that squeezes the last decimal.
- When the budget is nearly gone, stop trying new ideas and make sure the best submission is
  solid.

## Finishing

After `mlguy finish`, write what you learned into `PROJECT_MEMORY.md`: what worked, what did
not and why you think so, and what you would try with more budget. Keep it short and concrete
enough that someone picking the project up later does not repeat your dead ends.

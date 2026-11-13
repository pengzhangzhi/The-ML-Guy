---
name: research
description: Improve a model's score on a The-ML-Guy project, within its budget. Use when a project has been initialised with `mlguy init` and validated, and the user wants the model made better.
---

# Improving the model

Your job is to improve the project's score, within its budget. How you do
that is entirely your call.

## What is fixed

Read `task/brief.md` and `task/task.toml` first. The task, the data, the split, the metric and
the budget are settled and must not be edited — `mlguy` checks `task/` on every submission and
refuses if it changed. That includes `task/evaluate.py`, when the task has its own evaluator.

One more thing is fixed, the command that runs your model:

```
python infer.py --model model --input PATH --output PATH [the task's [infer] args]
```

`--input` is a split's inputs, in whatever form the task stores them. `--output` is a file or a
folder, in whatever form `task/evaluate.py` reads; `[infer] args` in `task.toml`, if any, are
passed as `--key value`. Read the evaluator: it is the only thing that decides what a good output
looks like, and it refuses outputs it cannot score. `mlguy` and the project's page both call
`infer.py`, so it must keep working.

`infer.py` is run from a copy of the project holding only `infer.py`, `src/` and `model/` — and
the user runs the finished model the same way, out of `final/`. Neither has a `task/`, so
anything the model needs when it predicts belongs in `model/`.

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
mlguy finish                       # when you judge the work is done (--experiment N to pick)
```

`mlguy submit` runs your model, scores what it produces, records the experiment with a snapshot of the
code and model, and tells you the score and the remaining budget. It refuses once the
submissions are spent. Any other limit in `[budget]` — hours, a deadline, GPU-hours, money — is
the user's and is yours to keep: `mlguy` shows it to you and prints it in the report, and
stops nothing. A crash or outputs the evaluator refuses are recorded as an error, not as a score of zero,
but it still uses up a submission, so test `infer.py` before you submit.

`mlguy submit` runs `infer.py` on the machine you run it on. If the model needs a GPU or a
scheduler job to predict, run `mlguy submit` inside such a job (with Slurm, `srun ... mlguy
submit --note "..."`); the "Environment" section of `PROJECT_MEMORY.md` says what this machine
offers, and the `environment` skill covers finding out if it is missing.

Each submission gets one score, on the same validation set every time; that score is the
result of the run. A small gain on a finite validation set may be luck, so judge it as you
would any measurement.

**Choosing which submission to ship is yours.** By default `mlguy finish` takes the highest
validation score; `mlguy finish --experiment N --reason "..."` takes another, and the reason goes
in the report.

Validation is the shared yardstick, and its reference is in the project only so that it can be
scored. Read whatever `submit` reports, including any breakdown the evaluator gives. Make your
own choices (hyperparameters, which idea to submit) on the training data, by cross-validation
or however you like. Computing your own numbers against the validation reference outside `submit`
makes every recorded score optimistic.

The final check belongs to `mlguy finish`: once, at the end, on whichever submission scored
best. Do not go looking for the test split, and do not tune a model to an evaluator's blind spots: a score
won that way will not survive the final check or the user.

## Finishing

After `mlguy finish`, write what you learned into `PROJECT_MEMORY.md`: what worked, what did
not and why you think so, and what you would try with more budget. Keep it short and concrete
enough that someone picking the project up later does not repeat your dead ends.

When you tell the user where things stand, during the search or at the end, follow the
`writing` skill: the score and what it means for them first, then what you tried.

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

## Use external research when it helps

Before substantial modeling, consult `RESEARCH_CONTEXT.md` if present and use the `research-scout` skill to fill important gaps in approaches, implementation, or domain knowledge. During the loop, invoke it again when a targeted search could inform the next experiment. Scout provides evidence-backed, implementation-ready ideas; you decide what to try and write the code. Do not alter the frozen task or evaluator.

## The loop

```bash
mlguy status                       # what has been tried, what is best, what budget is left
# ... change whatever you want, retrain into model/ ...
mlguy submit --note "one line on what you changed and why"
mlguy finish                       # when you judge the work is done (--experiment N to pick)
```

`mlguy submit` runs your model, scores what it produces, records the experiment with a snapshot of the
code and model, and tells you the score, the time left and how many submissions there have
been. Submit whatever is worth having on the record: submissions are counted and reported, not
capped, unless the user set `max_submissions`. It refuses once the search's `hours` are up.
Any other limit in `[budget]` — a deadline, GPU-hours, money — is the user's and is yours to
keep: `mlguy` shows it to you and prints it in the report, and stops nothing. A crash or outputs
the evaluator refuses are recorded as an error, not as a score of zero, so test `infer.py`
before you submit.

`mlguy submit` runs `infer.py` on the machine you run it on. If the model needs a GPU or a
scheduler job to predict, run `mlguy submit` inside such a job (with Slurm, `srun ... mlguy
submit --note "..."`); the "Environment" section of `PROJECT_MEMORY.md` says what this machine
offers, and the `environment` skill covers finding out if it is missing.

Each submission gets one score, on the same validation set every time; that score is the
result of the run. A small gain on a finite validation set may be luck, so judge it as you
would any measurement.

**Choosing which submission to ship is yours, and the best score is only the default.** By
default `mlguy finish` takes the highest validation score; `mlguy finish --experiment N --reason
"..."` takes another, and the reason goes in the report. Take the default when it is right, but
look at the whole table before you do, and weigh whatever the brief makes relevant:

- **Does the gap mean anything?** On this many examples, two scores a hair apart are one
  measurement, not two. Prefer the simpler model when they are level.
- **What does it cost to run?** `mlguy status` reports, per submission, how long `infer.py` took
  over the validation split and how big `model/` is. A model that scores 0.002 better, is forty
  times larger and needs a GPU at prediction time is usually the wrong answer for someone who
  will run it on a laptop. Read the brief for which of those they actually care about.
- **Is the better score real, or is it the metric's blind spot?** A frozen metric cannot see
  everything that matters. If the top model wins by doing something the brief would call wrong,
  say so and take the other one — this is what `--reason` is for, and it is the most valuable
  thing you can put in a report.
- **What will the user have to maintain?** Fewer dependencies, no bespoke inference environment,
  and code someone else can read are worth a little score.

There are no fixed penalties for complexity and no thresholds to clear. Make the judgement, and
write down why in `--reason` so the user can disagree with it.

Validation is the shared yardstick, and its reference is in the project only so that it can be
scored. Read whatever `submit` reports, including any breakdown the evaluator gives. Make your
own choices (hyperparameters, which idea to submit) on the training data, by cross-validation
or however you like. Computing your own numbers against the validation reference outside `submit`
makes every recorded score optimistic.

The budget is a ceiling, not a target. When you judge that more submissions would mostly fit
noise, stop, and tell the user why.

The final check belongs to `mlguy finish`: once, at the end, on whichever submission scored
best. Do not go looking for the test split, and do not tune a model to an evaluator's blind spots: a score
won that way will not survive the final check or the user.

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

When you tell the user where things stand, during the search or at the end, follow the
`writing` skill: the score and what it means for them first, then what you tried.

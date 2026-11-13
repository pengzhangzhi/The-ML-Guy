---
name: evaluation
description: Define what success means for a The-ML-Guy project and turn it into something a search can climb — elicit it from the user, translate it into a metric measured on a split, stress-test the translation, and write task/evaluate.py. Use while building a project, before `mlguy init` freezes it.
---

# Defining success

The evaluation is the definition of success. The search will climb exactly what it rewards, on
exactly the data it is measured on, and both are frozen at `mlguy init`. A good model judged by
the wrong evaluation is a wrong answer delivered with confidence.

**Who does what.** The user decides what success means. You ask, propose, and translate it into
something measurable, then try to break the translation. The user approves it.

## 1. Ask what success means

Ask about the use, not the metric. Ground each question in their data — show them rows — and
offer the answer you would guess.

- **What will the outputs be used for?** The decision or action they feed: which candidates to
  make, which cases to look at, a measurement to replace, designs to build.
- **What will new inputs be like, next to the data they have?** New families, new patients, a
  later date, another lab. This decides what the score has to be measured on.
- **How would they judge a handful of outputs themselves,** and what do they trust to do it today?
- **What would convince them it works,** and what failure must they never see?

If they name a metric, a split or a model, take it as a requirement. Check it against their
answers, and say so plainly if the two disagree.

Write the questions, and the translation below, following the `writing` skill.

## 2. Translate it

Write the objective in one line, then show where each part came from:

> Maximise Spearman correlation on variants from enzyme families not seen in training.

| Part | From what they said |
| --- | --- |
| Spearman | "we test the top 50 the model ranks" — the order matters, not the values |
| families not seen in training | "our next round is new families" |

The translation has three parts, and each has to be something the user can check:

- **One primary number and its direction.** It is what the search climbs.
- **The split it is measured on.** The test split must look like the use in the second answer,
  and the validation split — the one the search climbs — must ask the same question as the test.
  Examples that are related (the same subject, family, batch, source, or near-copies) go to one
  side together, whatever grouped them. Nothing is scored on the test split until `mlguy finish`.
  Whatever the split cannot avoid, say in `validation_note`; it is quoted in the report.
- **What else is reported or refused.** Secondary metrics for what the primary cannot see, and
  refusals for outputs that cannot be scored.

Refusals also apply at `mlguy finish`, and a refusal there is recorded as the result, so set any
threshold with the margin real data needs between validation and test.

## 3. Try to break it

Answer each with numbers from `mlguy score --outputs PATH`, which scores any outputs on the
validation split and records nothing.

1. **Can it be won without solving the problem?** Score degenerate outputs: a constant, a
   shuffle, copies from training, one good output repeated, the shortcut this problem invites.
   None should come near a good score.
2. **Does it tell good from bad?** The best outputs available — the true answers, or something
   known to be good — should score at or near the top.
3. **Can it see a real difference?** Score two plausible models: the obvious default and
   something the field trusts. Is the gap bigger than resampling this many examples would move
   it? If not, every submission will be climbing noise.
4. **Is there headroom?** Score the simplest reasonable model now. If it is already near the
   ceiling, the search has nothing to find.
5. **Does it reward what they asked for, and only that?** Imagine the model that games it, and
   ask whether the user would accept that model. If not, add a secondary metric or a refusal.

It also runs on every submission, so its time is part of the budget. Any failure here is cheap
to fix now and impossible to fix after `mlguy init`: change the metric, the split, or both. Keep
the numbers for the brief; the audit will repeat the checks without having seen yours.

## 4. Write it

Copy `reference/evaluate.py` beside this skill to `task/evaluate.py`. It is the contract with one
function left for you:

- It is called as `evaluate.py --outputs PATH [--reference PATH] [--inputs PATH] --split
  validation|final --result FILE`. `--outputs` is whatever `infer.py` wrote. `--reference` and
  `--inputs` are the split's, passed when `task.toml` declares them; for the final check they are
  the held-back test ones, so use them rather than paths of your own.
- It writes `{"metrics": {name: number}}` with the primary finite; others may be NaN.
- It refuses what it cannot score, with an error. Never a zero.
- It writes nothing but `--result`. It runs with `task/` as its working directory, which is
  frozen.
- If `infer.py` needs fixed arguments — a number of samples, a seed, a threshold — put them in
  `[infer] args` in `task.toml`. The evaluator reads them from there and enforces what they imply.
- If scoring needs another environment, have the script start it, and set
  `[evaluator] timeout_minutes`.

Training code may import it, to tune against the same measure.

## Worked cases

`cases/` holds what earlier projects learned about particular shapes of problem, each with a
filled-in evaluator. Open one only if your problem looks like it. They are examples, not rules,
and a problem that fits none of them is expected: the four steps above are the method.

- `cases/matched-outputs.md`: one output per example, compared with a known answer.
- `cases/sampling.md`: the model draws samples, judged without an answer key.

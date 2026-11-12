# What came out

Four runs: two with Claude Code, two with Codex, spread across framework versions. This is the
most-repeated problem in the gallery, and the runs agree.

| Run | Agent | Primary metric (its own name for it) | Final score | The prior-admissions rule | Submissions |
| --- | --- | --- | --- | --- | --- |
| 1 | Claude Code | `recall_top20` | 0.3895 | ~0.34 | 2 of 10 |
| 2 | Claude Code | `readmissions_caught_top20` | 0.3838 | 0.34 | 2 of 10 |
| 3 | Codex | `recall_at_20pct` | 0.3800 | 0.347 | 3 of 10 |
| 4 | Codex | `recall_at_20pct` | 0.3881 | 0.342 | 4 of 10 |

The four metrics are the same idea under four names, computed on splits the agents drew
themselves, so the scores are **close but not strictly comparable**. What is comparable is each
run's gap over the prior-admissions rule it measured itself: **+4 to +5 points of recall**, every
time.

In the terms the scientist asked for (run 4, the most legibly reported): at the same call
capacity of 3,889 discharges, the model found 820 of 2,113 recorded 30-day returns against the
rule's 722 — **98 more returns for the same number of calls.**

## What they built

Nothing exotic, and all four converged on the same family: gradient-boosted trees (LightGBM) or a
logistic-regression blend over the tabular fields — prior visit counts, age, ICD-9 groups, payer,
specialty, discharge disposition, medication changes. CPU only, minutes per run.

Every run stopped early — 2, 2, 3 and 4 submissions out of a budget of 10 — on the grounds that
further attempts would be fitting noise. None was told to.

## What not to believe

- **Encounter number stands in for time,** because the file has no discharge dates. The
  chronological split is therefore approximate.
- **"Readmitted" means recorded in this dataset.** A patient readmitted to a hospital outside the
  group is invisible, and counts as a negative.
- **Encounters, not patients.** A patient can appear several times in the same score.
- **Calling someone does not prevent a readmission.** Every run says this explicitly: the number
  describes who the list finds, not the effect of acting on it.
- **The rule comparison is retrospective,** computed after model selection, though the rule
  itself was defined before the test labels were opened.
- The data is 1999–2008 and from one hospital group. It says nothing about a different hospital
  today.

## Bugs this problem found in the framework

As the first problem ever run by a fresh agent, it found the most structural ones:

- The built-in metrics rejected string and multi-class labels outright, and `f1`/`mcc` were
  binary-only. (This chain of findings ended with built-in metrics being removed altogether —
  every task now carries its own evaluator.)
- `task/__pycache__` broke the task fingerprint, so a submission was refused for a change nobody
  made.
- The report echoed internal config (`higher_is_better`) and claimed the test was scored "after
  the budget was spent" when the agent had stopped early.
- The skills said nothing about whether an agent was allowed to stop early, which all four runs
  wanted to do.

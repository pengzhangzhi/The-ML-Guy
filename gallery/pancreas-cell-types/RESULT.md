# What came out

Two runs, both Claude Code, the second after a batch of framework fixes.

| Run | Metric (its own name) | Held out for the final score | Final score | Submissions |
| --- | --- | --- | --- | --- |
| 1 | `type_balanced_recall` | Muraro (CEL-Seq2), 2,285 cells | 0.9253 | 3 of 8 |
| 2 | `islet_f1` | Baron inDrop donors 1 and 4, 3,237 cells | 0.9397 | 2 of 8 |

The two runs **chose different splits from the same answers**, which is worth noticing. Run 1
held out Muraro; run 2 held out the droplet study and reserved two of its four donors for the
final score. Both are defensible readings of "a new sample from our own lab", the scientist
approved each in turn, and the numbers are not comparable across them. The split is part of what
the agent builds.

Run 2's split, as the brief recorded it:

| Set | Cells | From | Used for |
| --- | --- | --- | --- |
| Train | 7,813 | five plate-based studies | Fitting |
| Validation | 5,325 | Baron inDrop, donors 2 and 3 | Feedback on every attempt |
| Test | 3,237 | Baron inDrop, donors 1 and 4 | One final score |

Roughly 0.93–0.94 macro-averaged across 13 cell types, on a technology the model never saw, with
the rare endocrine types the lab cares about scored as heavily as the common ones.

## What they built

Normalise counts, select genes, fit a linear classifier — then small improvements on top. Both
runs stopped after 2–3 submissions out of 8, judging that more would be noise. Minutes of CPU
training per attempt.

## What not to believe

- **One held-out study is one sample of "a new lab".** It is a far better test than a random
  split, but it is still n=1 on the thing that actually varies.
- **The labels are the atlas's labels,** not ground truth. Where the original annotators were
  wrong or inconsistent between studies, the model is scored against that.
- **Epsilon has 14 cells in the test set.** Its contribution to a macro average is extremely
  noisy, and it weighs as much as a type with 500. That was the scientist's explicit choice, and
  it makes the headline number jumpier than it looks.
- **Contamination was not modelled.** The scientist mentioned exocrine and stellate contamination
  in their own samples; the atlas's cells are already annotated and reasonably clean.

## Bugs this problem found in the framework

- The judge crashed when a model predicted a class that was absent from the split — which happens
  constantly with 13 types and 14 epsilon cells.
- A weighted/balanced per-item metric produced a confidence interval above 1.0.
- Multi-class and string labels were rejected by the built-in metrics (fixed here, then removed
  altogether later in favour of per-task evaluators).
- No test predictions were available when writing the report, so the agent could not describe
  what the model got wrong.

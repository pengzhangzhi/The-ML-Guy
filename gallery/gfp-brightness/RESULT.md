# What came out

Two runs on the same problem, one with Claude Code and one with Codex. **Their scores are not
comparable** — each wrote its own evaluator and drew its own split from the same answers, so one
measures "precision in the top 100 of a 3,000-candidate list" and the other "precision in the top
3%".

| Run | Agent | Metric (its own name) | Final score | Submissions |
| --- | --- | --- | --- | --- |
| 1 | Claude Code | `precision_at_100` | 0.9300 | 5 of 12 |
| 2 | Codex | `precision_at_top3pct` | 0.5647 | 5 of 8 |

Read each report, not the table.

## Run 1, in the scientist's terms

> If you give this model a few thousand designs carrying 6 to 12 mutations and synthesise the 100
> it ranks highest, about 93 of them will be bright.

Measured on 4,655 variants held back to the end and scored once, against the alternatives on the
same 4,655:

| How you choose 100 | How many are bright |
| --- | --- |
| At random | 5 or 6 |
| **Fewest mutations first** (current practice) | **14** |
| **This model** | **93** |

About a 7× improvement on current practice. The baseline that current practice represents was
measured, not asserted, because the agent asked for it during briefing.

## The protein language model did not help

The scientist's one methodological prior was "I've heard protein language models help with this."
The run tested it properly and reported the negative:

- ESM-2 3B's zero-shot masked-marginal score **ranks at chance on this problem** (AUROC 0.542;
  0.518 for the 650M model).
- Adding it as features to the winning blend moved nothing (internal AP 0.8774 against 0.8777).
- The agent submitted that experiment anyway, explicitly so the report would carry a real
  validation number for the question the scientist had asked.

The model that won is a rank-average of a ridge regression and a logistic regression on
substitution indicators plus mutation count. No language model, no GPU needed at prediction time.
Being able to tell a scientist "the method you had heard about does not help here, and here is the
number" is a good outcome, and it cost one submission out of twelve.

## The audit found a real leak, twice

Both runs' independent auditors found the same thing without being told to look: the scientist's
own raw CSV sat beside the project and joined to the inputs by sequence, so hiding the test labels
achieved nothing. On run 1 the auditor **recovered every validation answer and scored a perfect
1.0** to prove it. The raw file was moved before the freeze and the report says where it went.

This is the finding that justified the audit skill existing.

## What not to believe

- **Both holdouts are variants from the same deep mutational scan,** split by mutation count. Real
  computational designs may not look like this scan's sequences, and that shift is not measured.
  Run 2's report carries this as an explicit warning on the score table.
- **Brightness is one assay on one protein.** Nothing here transfers to a different property or a
  different protein.
- **The 7× over current practice** compares against picking by fewest mutations, which is what the
  scientist said they do. It is not a comparison against a serious published predictor.
- The model never sees mutation count at prediction time, by design, so it cannot be accused of
  relearning the heuristic — but that also means it is solving a slightly harder problem than a
  model that could.

## Bugs this problem found in the framework

- **The big one:** `submit` scored the working folder, but `finish` scores the submission's
  snapshot. Run 1's winning experiment resolved `task/data` relative to its own file, which works
  in the project and fails inside a snapshot, so it passed every submission and then died at the
  final check with `FileNotFoundError`. It had to spend another submission resubmitting the
  identical model with a path fix. `validate` and `submit` now snapshot first and score the
  snapshot, so this fails on the first submission instead of at the end.
- Compute discovery was validated here: both runs found the cluster's partitions and asked only
  which account to charge.

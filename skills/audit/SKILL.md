---
name: audit
description: Try to break a The-ML-Guy project's split and evaluator before they are frozen, as an outsider with no stake in the result. Use when a project has a drafted task/ (task.toml, data, evaluate.py) and has not yet been through `mlguy init`, or when asked to audit a project's rubric.
---

# Auditing the rubric

Another agent built this project's split and wrote its evaluator, and is about to hill-climb on
them. Everything it does from here is judged by those two things, so a flaw in them is a flaw in
every result. Your job is to find the flaws before they are frozen. You have not seen the
conversation with the user, and that is the point: judge what is on disk.

Read `task/brief.md`, `task/task.toml`, `task/evaluate.py`, the script that prepared
`task/data`, and the data itself. Do not change any of it; report. Do not train models beyond
what a check needs, and never score anything against the test labels: count and compare them,
nothing more.

## The evaluator

The search will optimise exactly what it rewards. Run it with
`mlguy score --outputs FILE` (it scores validation) on outputs you construct, and record each
score:

- **Good:** the true labels written as outputs, where that makes sense. It should score at or
  near the best possible value.
- **Degenerate:** a constant, the majority class, shuffled labels, random values, training
  labels copied over by any shared key, one good output repeated (for generation). None of
  these should come anywhere near "good".
- **Malformed:** a missing id, a repeated id, an id that is not in the split, a missing column,
  a NaN. Each should be refused with an error, not scored, and certainly not scored as zero.
- **Reordered:** the same outputs in another row order must score the same.

Then read the code: is the primary metric the one the brief describes, computed the way the
brief says, in the stated direction? Ties, rounding, cut-offs and averaging over classes or
groups are where evaluators quietly differ from their description.

## The split

- **Leakage between splits:** the same or near-identical examples in two splits; related rows
  (the same patient, donor, family, sequence cluster, slide) spread across them when the brief
  says new cases will be unrelated.
- **Leakage inside the inputs:** columns that encode the answer or the future (a discharge code
  that implies the outcome, a timestamp after the event, an identifier that links a test row to a
  labelled training row).
- **Does validation ask the test's question?** Compare how they were drawn and what they
  contain. If they differ, is `validation_matches_test = false` set, and is the reason given?
- **Does the test match the use?** The brief says what new inputs will look like. Would a high
  score on this test set mean the model works there?
- **Sizes:** enough examples, and enough of each class or group the metric depends on, for the
  score to mean something.

## What to write

Write `task/audit.md`:

1. **Verdict,** in one line: either nothing found that should stop the freeze, or what must be
   fixed first.
2. **Findings,** most serious first, each with the evidence (the command, the number, the
   example rows) and what would fix it.
3. **The evaluator table:** every output you tried, and its score or error.
4. **For the user,** a short paragraph in plain language: what was checked, and what they
   should know before approving.

Then end with a short message that says where the audit is and what its verdict is. Be
specific and be fair: a finding without evidence is noise, and a clean audit should say so
plainly.

# The model draws samples, judged without an answer key

Worked evaluator: `evaluate_sampling.py`, which reads the sample count from `[infer] args`,
refuses a wrong count or repeated samples, and averages over every sample.

What earlier projects of this shape learned:

- **Score every sample, not the best one.** The best of twenty scores far better than twenty do,
  so a maximum over samples rewards drawing more rather than generating better.
- **Fix the count and enforce it.** Put `n` and `seed` in `[infer] args`, and a larger `n` and a
  fresh seed in `final_args`. The evaluator reads them and refuses any other count, so a
  submission cannot quietly be the best few of many.
- **The judge is the user's.** Ask how they would judge a batch by hand, and use what they
  already trust to do it: a folding model, a classifier, a simulation, a comparison with real
  data. It often needs its own environment; have the evaluator start it.
- **Fold validity, diversity and novelty into the score, or refuse what fails them.** Otherwise
  one good output repeated, or a copy of training data, wins.

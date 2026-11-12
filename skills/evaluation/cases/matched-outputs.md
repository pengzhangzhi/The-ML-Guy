# One output per example, compared with a known answer

Worked evaluator: `evaluate_matched.py`, which matches `id, prediction` to `id, target` and
refuses missing, repeated or unknown ids.

What earlier projects of this shape learned:

- **The use picks the family of metric.** Ranking candidates to test calls for a rank metric, or
  for precision among the top k they will actually test. Needing the value itself calls for an
  error in its units. A decision with a threshold calls for the cost of each kind of mistake.
- **A metric that moves with the base rate misleads when the rate differs between splits.** Plain
  accuracy on a validation set that is 42% positive, against a test set that is 50%, ranks models
  partly by how they guess the rate. A balanced measure is safer; if a base-rate measure is kept
  to compare with published numbers, say why in the brief.
- **Rare classes the user cares about need their own weight.** An average over examples is
  dominated by the common classes; averaging per class, or weighting the classes the user named,
  makes the score see what they see.
- **Use a well-tested library for standard metrics.** Ties, empty classes and averaging are where
  hand-written ones quietly differ from their description.

# Pinning down the problem

The agent proposed the split before asking anything else: hold out a whole study, so the test
asks the question the lab will ask.

**Your own samples are droplet-based — should the test be too?**

> Yes. We run 10x Chromium 3' v3. Testing on a study from a different platform than ours would
> flatter the number.

So it trained on the five plate-based studies and tested on the one droplet study, which the
model never saw.

**Should every cell type count equally, or should common ones dominate?**

> Every type equally. Delta, gamma and epsilon are the ones we struggle to annotate by hand.

That rules out plain accuracy, which alpha and beta would dominate. The metric became a
macro-average over types, so epsilon's 14 test cells weigh as much as alpha's 520.

**May the model say "none of these"?**

> Always picking a known type is fine, as long as there's a confidence so we can review the
> uncertain cells.

**What about the ambiguous labels?**

> Drop the T cells, there are too few. If activated and quiescent stellate get mixed up with each
> other, that matters much less to us than mistakes between the endocrine types.

So the two stellate states are scored as one group, and still reported separately.

**Budget?**

> About 8 attempts.

## Approval

> Testing on the droplet study makes sense for us. The examples of what the scores mean were
> very helpful. Please go ahead.

# Automatic islet cell-type annotation

## The request

> I run an islet biology lab. Every time we sequence a new pancreas sample we annotate the cell
> types by hand, which is slow and inconsistent between people. I'd like a model that does it
> automatically.
>
> pancreas.h5ad is a public human pancreas single-cell atlas pooled from several published
> studies, with cell types already annotated. Our own samples are 10x Chromium 3' v3 on isolated
> islets from organ donors, healthy and T2D, and there's always some exocrine and stellate
> contamination.
>
> The types we struggle with most are delta, gamma and epsilon — they're rare, and they're the
> ones we actually care about getting right. CPU is fine.
>
> Can you build that with The-ML-Guy?

## Pinning down the problem

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

### Approval

> Testing on the droplet study makes sense for us. The examples of what the scores mean were
> very helpful. Please go ahead.

## What you get

**A classifier that annotates 13 islet cell types on a platform it has never seen, getting the
rare endocrine types right as often as the common ones.**

Scored once on 3,237 cells from a held-back droplet study:

| | |
| --- | --- |
| Macro-averaged F1 across 13 types | **0.94** |
| Cells labelled correctly | 95% |
| Rare types (delta, gamma, epsilon) | scored equally with alpha and beta, by design |

Every cell also gets a confidence, so the uncertain ones can be pulled for manual review instead
of being silently wrong.

**What it took:** 2 of 8 submissions, CPU only, about a minute of training per attempt. The agent
stopped early rather than spend the rest of the budget on noise.

# What the agent asked, and what it was told

The answers are quoted as the scientist sent them. The questions are condensed from the agent's
own longer versions, each of which carried a proposed answer.

The agent opened by proposing the split itself — hold out a whole study — and the reply confirms
it before answering anything else.

## The five questions

> Thanks, that matches what I know about these datasets. Holding out whole studies makes sense to
> me.

**1. What will your own samples actually be?**

> We run 10x Chromium 3' (v3) on isolated islets from organ donors, both healthy and T2D.
> There's always some exocrine and stellate contamination.

This decided the split. The agent trained on the five **plate-based** studies and tested on the
one **droplet** study, because the lab's own platform is droplet — so the test set asks the
question the lab will ask.

**2. Should every cell type count equally?**

> Yes, every type should count equally. Delta, gamma and epsilon are exactly the ones we struggle
> to annotate by hand.

The metric follows directly: macro-averaged F1 over types, so epsilon (14 cells in the test set)
weighs as much as alpha (520). A plain accuracy would have been dominated by alpha and beta and
would have scored well while failing at exactly the job the lab wanted done.

**3. May the model say "none of these"?**

> Always picking one of the known types is fine, as long as there's a confidence so we can review
> the uncertain cells.

**4. What about the rare and ambiguous labels?**

> Dropping the T cells is fine. Keep the rest as they are. If activated and quiescent stellate get
> mixed up with each other, that matters much less to us than mistakes between the endocrine types.

So activated and quiescent stellate are scored as **one** group — confusing the two costs
nothing in the headline number, and they are still reported separately. A domain judgement that
no amount of looking at the data would have produced.

**5. What form will the data arrive in, and what is the budget?**

> We get Cell Ranger output and can turn it into an `.h5ad` of raw counts with gene symbols.
> About 8 attempts is fine.

## The approval

> Approved. Testing on the droplet study makes sense for us, and treating the two stellate states
> as one group is fine. The examples of what the scores mean were very helpful. Please go ahead.

Both runs of this problem explained the metric by worked example — "a score of 0.90 means roughly
that, for a typical type, 9 in 10 of its cells are found and 9 in 10 of the cells given that
label really are that type" — and both times the scientist singled that out as the thing that
made approval possible. The build skill now asks for approval by example because of this.

## What to take from this

- **Say what your own instrument is.** One sentence about the platform ("10x Chromium 3' v3")
  determined the entire split, because it told the agent which study to hold out.
- **Say which mistakes you don't care about.** "Stellate states mixing up matters much less than
  endocrine confusion" removed a whole category of false failure from the score.
- **Ask for the metric as an example, not as a formula.** If you cannot tell what a 0.90 would
  mean for your work, the rubric is not ready to freeze.

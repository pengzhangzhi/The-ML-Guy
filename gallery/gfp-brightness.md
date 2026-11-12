# Which GFP designs will still be bright

## The request

> I'm a protein engineer. We design new GFP variants computationally — a few thousand candidates
> at a time — and we can only afford to synthesise and test about a hundred of them. Right now we
> pick by fewest mutations, which is not much of a strategy.
>
> gfp_variants.csv has about 54,000 variants from a deep mutational scan, with measured log
> brightness and how many mutations each carries relative to the parent. The catch is that our
> designs usually carry more mutations than most of what was measured, so I don't want a model
> that only works close to the parent.
>
> I've heard protein language models help with this sort of thing, and we have GPUs on the
> cluster if you want them.
>
> Can you build it with The-ML-Guy?

## Pinning down the problem

The scientist had already named the hard part — designs carry more mutations than the training
data — so the agent proposed splitting by mutation count and asked five questions around it.

**Is splitting by mutation count right?**

> Yes. Our designs usually carry 6 to 12 mutations, so judging on that range matches what we do.

**How many candidates do you make, and how many can you test?**

> A few thousand designed, about a hundred synthesised. So score it on roughly the top hundred
> of a list that size.

That answer is the metric: precision in the top 100 of a ~3,000-candidate list.

**What counts as "still bright"?**

> Close to the parent. The parent is about 3.7, and below about 2.5 the protein is effectively
> dark. Set the line near 3.5.

**Substitutions only, or deletions too?**

> Substitutions only in our designs. Keep the deletion variants in training if they help, but
> score on substitutions, since that's what we'll actually make.

**Budget, and what would be good enough?**

> 8 to 12 attempts. Good enough would be clearly better than picking by fewest mutations, which
> is roughly what we do by hand today.

### The agent measured the bar before agreeing to it

Rather than take "better than current practice" as a vibe, it measured that practice (0.095) and
also measured what a skill-less rule scores if you try twelve times and keep the best (0.144,
inflated by the selecting). It proposed holding itself to 0.16.

> I hadn't thought about the fact that trying a dozen models and reporting the best inflates the
> number. Holding yourself to 0.16 before calling anything a win is the right instinct.

Approved, and frozen.

## What you get

**Give it a few thousand designs carrying 6 to 12 mutations, synthesise the 100 it ranks
highest, and about 93 of them will be bright.**

Scored once on 4,655 variants the model never saw:

| How you pick 100 to synthesise | How many are bright |
| --- | ---: |
| At random | 5 or 6 |
| Fewest mutations first (current practice) | 14 |
| **This model** | **93** |

About a 7× improvement on current practice — and "current practice" is a measured number, not an
assumed one, because the agent measured it during briefing.

**It also answered the question you asked in passing.** Protein language models did not help
here: ESM-2's zero-shot score ranks at chance on this problem (AUROC 0.54), and adding it as
features changed nothing. The agent spent one submission establishing that so the report could
say it with a number. The model that shipped is a ridge-plus-logistic blend on substitution
indicators — no language model, no GPU needed to run it.

**What it took:** 5 of 12 submissions. The agent stopped early.

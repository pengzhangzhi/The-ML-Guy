# Pinning down the problem

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

## The agent measured the bar before agreeing to it

Rather than take "better than current practice" as a vibe, it measured that practice (0.095) and
also measured what a skill-less rule scores if you try twelve times and keep the best (0.144,
inflated by the selecting). It proposed holding itself to 0.16.

> I hadn't thought about the fact that trying a dozen models and reporting the best inflates the
> number. Holding yourself to 0.16 before calling anything a win is the right instinct.

Approved, and frozen.

# What the agent asked, and what it was told

The answers are quoted as the scientist sent them. The questions are condensed from the agent's
longer versions, each carrying a proposed answer and a table of evidence.

This is the exchange where the **approval message is more important than the answers**. Read to
the bottom.

## The five questions

> That table is exactly the picture we have in the lab, thank you for laying it out.

**1. Is the split by mutation count right?**

> The split is right. Our designs usually carry somewhere between 6 and 12 mutations, so judging
> on the 6-to-15 set matches what we do. Keep the boundary where you have it.

**2. How many candidates do you make, and how many can you synthesise?**

> We design a few thousand candidates computationally and can only synthesise and test about a
> hundred, so score it on roughly the top hundred of a list that size.

That is the metric: precision in the top 100 of a ~3,000-candidate list. Again the scientist
described a bench constraint and the agent turned it into a measurement.

**3. What counts as "bright"?**

> "Close to the parent", please. The parent is about 3.7 and anything near it is usable; below
> about 2.5 the protein is effectively dark. Set the line near 3.5.

**4. Substitutions only, or also deletions?**

> Substitutions only in our designs. Keep the deletion variants in training if they help, but I'd
> rather the score came from substitution variants, since that is what we will actually make.

**5. Compute and budget.**

> Use the free preemptible queue, yes, and please do not touch the prepaid priority budget: it is
> shared with another lab. 8 to 12 attempts is fine. Good enough would be clearly better than
> picking by fewest mutations, which is roughly what we do by hand today.

## The approval, which is the interesting part

Before freezing, the agent had measured two things nobody asked it to measure: what the
scientist's current practice actually scores (0.095, i.e. picking by fewest mutations), and how
much a *skill-less* rule scores on average when you let it try twelve times and keep the best
(0.144 — because selecting the best of twelve inflates the number). It then proposed holding
itself to about 0.16 before calling anything a win.

> Approved, and thank you for the four examples — those are exactly the kind of designs we make,
> combinations of mutations we have each seen before but not together. That is our problem in one
> picture.
>
> Two things I'm glad you spelled out. First, moving my CSV: please leave it where you have put it,
> and say in the report that it is there. Second, the honest bar. I had not thought about the fact
> that trying a dozen models and reporting the best inflates the number, and 0.095 being "my
> current practice, measured" is a useful thing for me to know. Holding yourself to about 0.16
> before you call anything a win is the right instinct, and yes, please report the two steadier
> numbers beside it every time.
>
> Your assumptions are right: the floor readings are real darkness, not failed assays; dropping
> the deletion variants from the scoring is what I want; and withholding the mutation count is
> fair, since otherwise you would just be relearning what I already do by hand.

Three things in there worth noticing:

- **"Moving my CSV"** — the independent auditor had found that the scientist's own raw file sat
  beside the project and joined straight to the inputs, so hiding the test labels achieved
  nothing. Both runs of this problem found that leak independently. On one, the auditor recovered
  every validation answer and scored a perfect 1.0.
- **"Withholding the mutation count is fair"** — the model is not allowed to see the feature that
  encodes the scientist's existing heuristic, so the comparison against current practice means
  something.
- **The selection-inflation bar.** The agent set its own threshold above what noise plus twelve
  attempts would produce, and the scientist had not known that was a problem.

## What to take from this

- **State the shift.** "Our designs will carry more mutations than what we measured" is the one
  sentence that makes this problem honest. Without it a random split would have scored far higher
  and meant nothing.
- **Describe your bench constraints, not a metric.** "A few thousand designs, we can make a
  hundred" is more useful than any metric name, and the metric follows from it.
- **Say what you do today.** It became the comparison in the report and the bar for calling the
  project a success.
- **Keep your raw data out of the project folder,** or expect the audit to tell you that your
  held-out answers are sitting next to your model.

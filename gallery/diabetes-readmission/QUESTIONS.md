# What the agent asked, and what it was told

The answers are quoted as the scientist sent them. The questions are condensed: each arrived as a
paragraph with the agent's own proposed answer and its reasoning, which is why the reply opens
with "mostly yes to your guesses".

All four runs of this problem asked substantially the same five questions, in roughly this order.
That consistency is itself a result — the questions are a property of the problem, not of the
agent.

## The five questions

**1. How many patients can the team actually call, and what does "good" mean?**

> The team can manage roughly the riskiest 1 in 5 discharges. What matters to me is how many of
> the people who really come back within 30 days are in that group.

This answer *is* the metric: recall among the top 20% by predicted risk. Note that the scientist
never said "recall@20%" — they described a staffing constraint, and the agent turned it into a
metric.

**2. What will the model see in use, and how should the split be drawn?**

> Yes: future discharges, and many of them are people we've seen before. Keeping each patient on
> one side of the split and using the latest encounters as the test sounds right to me.

Two traps closed in one sentence: group the split by patient, and order it by time.

**3. Which encounters should be excluded, and what counts as readmitted?**

> Agreed: leave out deaths and hospice. Also leave out transfers to another acute hospital, since
> they don't go home from us. Keep the patients who left against medical advice. They are exactly
> the people we'd want to call. Only "<30" counts as readmitted.

The agent proposed the exclusions; the scientist added one (acute transfers) and *rejected* one
by keeping against-medical-advice discharges — the clinically loaded call, which only the domain
expert could make.

**4. Which fields may the model use?**

> Leave race out. Age, sex and insurance are fine.

**5. Budget, and what would count as good enough?**

> Up to 10 attempts is fine. I don't have a hard number for "good enough". If it does clearly
> better than a simple rule based on prior admissions, that would already be useful to us.

## The approval

> Approved. This is clear and it matches how we'd use it. Catching more than about 1 in 3
> readmissions in the top 20% would already help us. Please go ahead.

The bar arrives at approval time, not with the request — and it arrives as a comparison against
current practice ("a simple rule based on prior admissions") rather than as a number. Every run
then measured that rule and reported the model beside it.

## What to take from this

- **A four-sentence request is enough to start, not enough to freeze.** The agent's job was to
  turn it into something scoreable; nine-tenths of that is asking.
- **Name your current practice.** "Better than how we do it today" is a far more useful bar than
  a number, because it is measurable and it is the comparison you actually care about.
- **Answer the exclusion question carefully.** It is the one where a reasonable-sounding default
  (drop the against-medical-advice discharges as noisy) would have removed exactly the patients
  the tool exists to find.

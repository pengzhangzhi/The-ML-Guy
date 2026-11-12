# 30-day readmission at discharge

## The request

> I'm a hospitalist working on care transitions. Our care-coordination team can only call a
> fraction of the patients we discharge, and right now they pick by gut feel plus whoever has
> been in a lot recently.
>
> I want a model that flags, at discharge, which patients are likely to come back within 30
> days, so the team knows who to call. diabetic_data.csv has about 100k hospital encounters for
> diabetic patients from a group of US hospitals, with IDS_mapping.csv for the codes.
>
> Whatever you build needs to beat what we do today, which is roughly "call the people with the
> most prior admissions". CPU is fine, this shouldn't need a GPU.
>
> Can you build that with The-ML-Guy?

## Pinning down the problem

The agent opened the file first, then asked five questions. Each came with its own proposed
answer, so most of the replies are confirmations.

**How many patients can the team actually call?**

> Roughly the riskiest 1 in 5 discharges. What matters is how many of the people who really come
> back are in that group.

That answer is the metric: recall among the top 20% by predicted risk.

**What will the model see in use, and how should the data be split?**

> Future discharges, and many of them are people we've seen before. Keep each patient on one
> side of the split, and use the latest encounters as the test.

**Which encounters should be excluded, and what counts as readmitted?**

> Leave out deaths and hospice, and transfers to another acute hospital — they don't go home
> from us. Keep the patients who left against medical advice; they're exactly who we'd call.
> Only a return within 30 days counts.

**Which fields may the model use?**

> Leave race out. Age, sex and insurance are fine.

**Budget, and what would be good enough?**

> Up to 10 attempts. If it does clearly better than the prior-admissions rule, that's useful to us.

### Approval

> This matches how we'd use it. Catching more than about 1 in 3 readmissions in the top 20%
> would already help. Please go ahead.

## What you get

**A call list that finds 13% more returning patients than current practice, for the same number
of calls.**

Scored once on a held-back later period of 19,443 discharges:

| At the same call capacity of 3,889 patients | 30-day returns found | Share of all returns |
| --- | ---: | ---: |
| **This model** | **820** | **38.8%** |
| Prior-admissions rule (what the team does today) | 722 | 34.2% |

That is 98 more returning patients caught, with no extra calls. The report states the comparison
in those terms rather than as a metric, because that is the decision the team actually makes.

**What it took:** 4 of 10 submissions, CPU only, minutes of training per attempt. The agent
stopped early, judging that more attempts would be fitting noise.

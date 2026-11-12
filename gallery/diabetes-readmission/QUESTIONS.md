# Pinning down the problem

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

## Approval

> This matches how we'd use it. Catching more than about 1 in 3 readmissions in the top 20%
> would already help. Please go ahead.

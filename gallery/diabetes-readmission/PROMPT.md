# 30-day readmission at discharge — the request

A clinical table. The interesting part of this problem is not the modelling, it is that almost
every decision that matters — who to exclude, what counts as a readmission, which fields are
allowed, how the split is drawn — is invisible in the request and has to be asked for.

## What the scientist asked for

> Hi. I'm a hospitalist and I do research on care transitions. The file `diabetic_data.csv` (with
> `IDS_mapping.csv`) has hospital encounters for patients with diabetes from a group of US
> hospitals. I'd like a model that flags, at discharge, which patients are likely to be
> readmitted within 30 days, so our care-coordination team knows who to call. Can you build that
> for me with The-ML-Guy?

Four sentences. Everything in [QUESTIONS.md](QUESTIONS.md) had to be extracted from it.

## What else the agent was told

- Where to work, and to build its own Python environment there.
- The machine has 4 CPUs, shared, and no GPU. Each training run should take minutes, not hours.
- Not to modify the tool's repository, and not to look in any other trial's folder.
- To stop and ask whenever a skill said to, and not to invent the answers.

## Where the data came from

The UCI "Diabetes 130-US hospitals for years 1999–2008" dataset: 101,766 encounters,
`diabetic_data.csv` plus the `IDS_mapping.csv` code lookup. Not redistributed here.

Two things about this data shaped every run. It has **no discharge dates**, so encounter number
has to stand in for time if the split is to be chronological. And it has **repeat patients**, so
a random split leaks the same person across train and test.

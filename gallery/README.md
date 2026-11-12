# Gallery

Five problems that have been built end to end with The-ML-Guy, kept here as **what the scientist
asked for**, not as code.

Each folder holds three short files:

| File | What it is |
| --- | --- |
| `PROMPT.md` | The request, in the scientist's own words, and what else the agent was told |
| `QUESTIONS.md` | What the agent asked back before it would freeze anything, and the answers |
| `RESULT.md` | What it built, what it scored, what it cost, and what not to believe |

## Why the prompt is the thing worth keeping

The modelling is the agent's job and the code is different every time. Your job is the other half
of the contract: say what you want well enough that the rubric can be built from it. That is the
part with no reference material anywhere else, so it is what this folder collects.

`QUESTIONS.md` is the most useful file in each folder. The gap between the request and what the
agent still had to ask **is** the measure of how complete the request was. On
[gfp-brightness](gfp-brightness/), the agent asked five questions and one of them changed the
whole metric. On [proteinmpnn-solubility](proteinmpnn-solubility/), the answer to question 1 was
wrong, the agent measured it, and the scientist reversed themselves before anything was frozen.
Reading those exchanges will tell you more about writing a good brief than any checklist.

## The problems

| Problem | Shape of the data | Runs | What came out |
| --- | --- | --- | --- |
| [diabetes-readmission](diabetes-readmission/) | Clinical table, 101,766 encounters | 4 | Beats the prior-admissions rule by 4–5 points of recall, every run |
| [pancreas-cell-types](pancreas-cell-types/) | Single-cell matrix, 16,382 cells, 13 types | 2 | ~0.93 balanced across types on an unseen technology |
| [pathology-tiles](pathology-tiles/) | 100k image tiles, 9 classes | 1 | 0.88 weighted F1 on later patients; stain augmentation was the lever |
| [gfp-brightness](gfp-brightness/) | 54,000 protein variants, continuous target | 2 | Ranks designs well; both runs found the metric was the hard part |
| [proteinmpnn-solubility](proteinmpnn-solubility/) | 869,544 structures, **generative** | 2 | One agent beat the published weights; the other honestly did not |

## Read the scores with care

**Scores from different runs of the same problem are not comparable.** The evaluator is part of
what the agent builds, so two runs of one problem measure different things — on
`gfp-brightness` one run scored "precision in the top 100 of a 3,000-candidate list" and the
other "precision in the top 3%". Both are reasonable readings of the same request. Neither
number means anything about the other.

**These are not benchmark results.** Each problem was run once or twice, with a different agent
or after a framework change. Where two runs of one problem agree, that is worth something; where
they differ, read both `RESULT.md` files before concluding anything.

**The prompts were written by a stand-in.** No real scientist sent these. They were written to be
realistic — with the gaps, the vagueness and in one case the contradiction that real requests
carry — but they are cleaner and better organised than most first messages, and they answer
follow-up questions promptly and in order. Treat them as a target to aim at, not as field
samples.

**The data is not here.** Each `PROMPT.md` says where its data came from. Nothing in this folder
is redistributed.

## Not in this folder

`examples/amylase` and `examples/solubility` are runnable code examples with a `make_task.py` you
can execute. They were built directly rather than from a prompt, so they have no entry here.

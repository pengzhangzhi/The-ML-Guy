# Gallery

Five problems built end to end with The-ML-Guy, kept as the conversation rather than the code.
Each folder shows the same three things:

| File | |
| --- | --- |
| `PROMPT.md` | What the scientist sent to start it |
| `QUESTIONS.md` | What the agent asked back before anything was frozen, and the answers |
| `RESULT.md` | What came out |

If you have a problem of your own, read the prompt closest to it and write yours the same way.
You do not need to know any machine learning to write a good one — none of these name a model, a
metric or a loss. They say what the data is, what the answer will be used for, and what counts
as better than what you do today.

| Problem | Data | What came out |
| --- | --- | --- |
| [proteinmpnn-solubility](proteinmpnn-solubility/) | 869k protein structures, **generative** | Half as many greasy surface residues, folds just as well, beat the published model |
| [diabetes-readmission](diabetes-readmission/) | Clinical table, 100k encounters | Finds 13% more returning patients than current practice, same number of calls |
| [pancreas-cell-types](pancreas-cell-types/) | Single-cell atlas, 16k cells | 0.94 across 13 cell types on an unseen platform |
| [pathology-tiles](pathology-tiles/) | 100k H&E image tiles | 93% correct, and tells you the tumour–stroma ratio reads low |
| [gfp-brightness](gfp-brightness/) | 54k protein variants | 93 of your top 100 designs are bright, against 14 picking by hand |

## The questions are the interesting part

`QUESTIONS.md` is short in every folder, and it is where the project is actually decided. A few
of them are worth reading even if the problem is nothing like yours:

- In **proteinmpnn-solubility**, the scientist's first answer about their own metric was wrong.
  The agent measured it, showed the hole, and they changed their mind before anything froze.
- In **pathology-tiles**, the agent asked about some odd violet tiles it had noticed. The answer
  decided the whole project.
- In **gfp-brightness**, the agent measured what the lab currently does, so "better than what we
  do today" became a number instead of a feeling.

## Reading the numbers

These are real runs, not benchmarks. Each problem was built once or twice, and each result was
scored a single time on data held back from the start. The reports say what they do not show,
and so does each `RESULT.md` here — that honesty is the point of the tool, not a disclaimer on it.

The data is not in this folder. Each prompt says what it is.

## Starting your own

There is no example project to copy. `mlguy new my-project` scaffolds the skeleton, and the agent
fills it in with you — that conversation is what this folder is a record of. Two worked
evaluators sit in `skills/build-project/reference/` for the one part with a fixed shape.

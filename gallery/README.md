# Gallery

Five problems built end to end with The-ML-Guy, kept as the conversation rather than the code.
Each one is a single page with the same three parts: **the request** the scientist sent, **the
questions** the agent asked back before anything was frozen, and **what you get**.

If you have a problem of your own, read the prompt closest to it and write yours the same way.
Write it in whatever terms you think in. Most of these name no model, metric or loss; the
ProteinMPNN one names the folding model and the score it wants, and the agent took them as
requirements. What every good one says is what the data is, what the answer will be used for,
and what counts as better than what you do today.

| Problem | Data | What came out |
| --- | --- | --- |
| [proteinmpnn-solubility](proteinmpnn-solubility.md) | 869k protein structures, **generative** | Half as many greasy surface residues, folds just as well, beat the published model |
| [diabetes-readmission](diabetes-readmission.md) | Clinical table, 100k encounters | Finds 13% more returning patients than current practice, same number of calls |
| [pancreas-cell-types](pancreas-cell-types.md) | Single-cell atlas, 16k cells | 0.94 across 13 cell types on an unseen platform |
| [pathology-tiles](pathology-tiles.md) | 100k H&E image tiles | 93% correct, and tells you the tumour–stroma ratio reads low |
| [gfp-brightness](gfp-brightness.md) | 54k protein variants | 93 of your top 100 designs are bright, against 14 picking by hand |

## The questions are the interesting part

The middle section of each page is short, and it is where the project is actually decided. A few
of them are worth reading even if the problem is nothing like yours:

- In **proteinmpnn-solubility**, the scientist's first answer about their own metric was wrong.
  The agent measured it, showed the hole, and they changed their mind before anything froze.
- In **pathology-tiles**, the agent asked about some odd violet tiles it had noticed. The answer
  decided the whole project.
- In **gfp-brightness**, the agent measured what the lab currently does, so "better than what we
  do today" became a number instead of a feeling.

## Reading the numbers

These are real runs, not benchmarks. Each problem was built once or twice, and each result was
scored a single time on data held back from the start.

What is summarised here in a few lines arrives as a full `REPORT.md`, and the part worth
expecting is the part that argues against itself. The pathology report, having given its 93%,
goes on to work out that because stroma is under-called the tumour–stroma ratio reads low — a
true 30% comes out around 21% — and says to use it for ranking slides rather than as a number.
Nobody asked it for that. Every report here has a section like it.

The data is not in this folder. Each prompt says what it is.

## Starting your own

There is no example project to copy. `mlguy new my-project` scaffolds the skeleton, and the agent
fills it in with you — that conversation is what this folder is a record of. Two worked
evaluators sit in `skills/build-project/reference/` for the one part with a fixed shape.

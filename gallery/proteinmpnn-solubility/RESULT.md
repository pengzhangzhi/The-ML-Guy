# What came out

Two runs, two different agents, **opposite results on the same problem.** That is the most
useful thing in this folder: the rubric held in both directions.

| Run | Agent | Approach | Result |
| --- | --- | --- | --- |
| 1 | Claude Code | Fine-tune with an added surface-hydrophobic penalty | Halved exposed hydrophobics; beat the authors' soluble weights |
| 2 | Codex (gpt-6-sol) | Fine-tune on the soluble-filtered full manifest | Did not beat its own baseline; said so |

Both chose the same shape of solution — fine-tune the authors' general `v_48_020` weights rather
than train from scratch, which neither thought credible in 12 GPU-hours — and both used the
authors' released soluble weights as the external comparator without being told to.

## Run 1: it worked

Final check, 126 unseen backbones × 8 designs = 1,008 designs, scored once, 4 of 6 submissions
used:

| | this model | real natural proteins, same backbones |
| --- | --- | --- |
| water-hating residues on the outside, per design | **7.3** | 14.9 |
| share of exposed positions in A/F/I/L/M/V/W/Y | **10.5%** | 21.4% |
| designs that fold back onto the requested backbone | **98.6%** | 100% |

`soluble_fold_eight = 0.9123`.

Side by side on the 64 validation backbones, where all three models were run together:

| | this model | published ProteinMPNN | the authors' soluble weights |
| --- | --- | --- | --- |
| greasy residues on the outside, per design | **6.9** | 13.8 | — |
| share of exposed positions in A/F/I/L/M/V/W/Y | **8.1%** | 18.9% | 15.0% |
| folds back onto the requested backbone | 97.9% | 98.1% | 96.7% |
| the score | **0.899** | 0.795 | 0.822 |

It ran out of its GPU budget before it could repeat that three-way comparison on the final 126
backbones, so **the comparison against the published weights is a validation number and the
report labels it as one** rather than mixing it into the headline table. Read it that way.

### Two findings worth more than the score

**Filtering the training data did nothing.** Training on soluble-filtered monomers alone moved
the surface measure from 18.9% to 19.0%. The objective was the lever, not the data — which is
the opposite of what the request assumed, and the run says so.

**It refused its own best model.** Experiment 3 scored 0.9386 on validation against experiment
2's 0.8993, and the agent took experiment 2 via `finish --experiment 2 --reason`. Its reason, in
the report:

> The metric rewards stripping the surface and has no way to see the cost: experiment 3's designs
> put 3.3% of their exposed positions in A/F/I/L/M/V/W/Y against nature's 20.9%, raise surface
> charge to 55.7% against nature's 35.7%, and all but eliminate surface alanine (0.5% against
> 6.7%). The only guard against that is a TM 0.8 line from a single-sequence structure predictor,
> which is too weak to justify shipping surfaces that far outside the natural distribution.

That is a model gaming a blind spot in a frozen metric, caught by the agent that was being scored
by it, with the higher number left on the table and the alternative kept in the report as a dial.
Of everything these trials produced, this is the strongest evidence that
`finish --experiment N --reason` earns its place.

## Run 2: it did not work, and the report says so

Three submissions, 2 counted against a budget of 2. All of them scored at or below the baseline:

| # | What was tried | joint_pass_rate (validation) |
| --- | --- | --- |
| 0 | the authors' unchanged soluble weights (baseline) | **0.8281** |
| 1 | 12,000-step fine-tune on the soluble-filtered manifest | 0.8047 |
| 2 | continued to 90,000 steps, a full pass over 308,804 chains | 0.7969 |

It finished with experiment 2 — its own full-pass model, not the better-scoring baseline — on the
grounds that the request asked for a retrainable model trained on data the lab controls, and
reported the baseline's higher validation score beside it.

On the final check its model scored `joint_pass_rate = 0.8438` against 0.7969 for the authors'
comparator on the same 16 backbones. It refused to call that a win:

> This six-design difference is small relative to variation among the 16 backbones: resampling
> backbones gives a 95% bootstrap interval of **-3.9 to +14.8 percentage points** for our minus
> their joint pass rate. It does not establish that ours is better on new backbones.

It also reported the wider-patch measure, on which it **loses** 36/128 to 57/128, and noted that
128 designs over 16 backbones are not 128 independent tests.

## What it cost

| | Run 1 | Run 2 |
| --- | --- | --- |
| GPU-hours | ~12 (the agreed ceiling) | 1.66 |
| Submissions | 4 of 6 | 2 of 2 |
| Hardware | A100 80GB, free tier | A100 80GB, free tier |

Neither touched the prepaid shared GPU budget. Run 2 was killed mid-training by a usage limit on
the agent's own account and resumed from its checkpoint four hours later without losing the
project.

## What not to believe

- **The held-out backbones are natural PDB structures, not designed ones.** Nobody tested this on
  RFdiffusion output, which is what the designs would actually be. Both reports say so.
- **Nothing was expressed.** "Soluble" here means a geometric estimate of exposed hydrophobic
  patches, not a solubility assay, an aggregation measurement or an expression yield.
- **ESMFold is the only check that a design folds,** and it is a single-sequence predictor being
  asked about sequences unlike anything in its training distribution.
- **ProteinMPNN's initialisation and ESMFold have both seen the PDB,** so "held out" is held out
  of *this* fine-tune, not of everything upstream of it.
- Run 1's comparison against the published weights is on validation, not on the final set.

## Bugs this problem found in the framework

Being the first generative problem run in anger, it found more than any other:

- `validate` and `submit` scored the working folder while `finish` scored the submission snapshot,
  so an `infer.py` that read from `task/` passed every submission and then failed at the final
  check with the budget spent.
- A generative project whose held-back test set had gone missing was silently scored on
  *validation* and the number reported as final.
- Generation ran no split checks at all, then printed "split checks: nothing raised".
- `mlguy score` on a hand-made file with one sample per condition flattered the evaluator in
  silence.
- `finish --experiment N` reports called the chosen model "the best submission" when it wasn't,
  and showed four of the evaluator's nine metrics — dropping the one that disagreed.
- `mlguy status` still said "run `mlguy finish`" after the final score had been taken.

All fixed.

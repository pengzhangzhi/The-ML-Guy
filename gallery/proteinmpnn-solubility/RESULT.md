# What you get

**A ProteinMPNN that puts half as many water-hating residues on the outside of a design, and
still folds.**

Scored once on 126 backbones it had never seen, 8 designs each:

| | this model | real natural proteins, same backbones |
| --- | --- | --- |
| water-hating residues on the outside, per design | **7.3** | 14.9 |
| designs that fold back onto the requested backbone | **98.6%** | 100% |
| mean self-consistency TM-score | 0.95 | — |

Run side by side on the validation backbones, it also beat the authors' own published soluble
weights (0.899 against 0.822) — which it was never told about, and found on its own as the
comparator worth measuring against.

**It refused its own best model.** One experiment scored higher but got there by stripping the
surface far outside anything natural — surface charge at 56% against nature's 36%, surface
alanine all but gone. The frozen metric could not see that; the agent could, and took the
moderate model instead, with the reason written into the report.

**What it took:** 4 of 6 submissions, about 12 GPU-hours on one A100.


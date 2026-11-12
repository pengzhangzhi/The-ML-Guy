# ProteinMPNN for soluble designs

## The request

> Current ProteinMPNN tends to design insoluble protein sequences that aggregate or are unable
> to express. I wanna develop a version of ProteinMPNN that can design soluble proteins; given
> the same backbone, it should put fewer water-hating residues on the surface, while folding into
> the backbone we asked for.
>
> I'll give u the original ProteinMPNN train data (the data link), here below, a CSV listing the
> PDB entries that should be left out when you want soluble-only training: membrane proteins and
> the like.
>
> For the evaluation, use ESMFold to fold designed sequences back and compare them with input
> backbones, and see the self-consistency TM-score and more soluble residues on the surface. Use
> the H200 GPUs on the cluster.
>
> Can you build this with The-ML-Guy?

## Pinning down the problem

The agent read the data and measured natural proteins as a yardstick — 25.8% of their surface
positions hold a greasy residue — then asked five questions.

**Which residues count as "water-hating"?**

> The big oily side chains: F, I, L, M, V, W. Alanine I don't worry about, it's small and natural
> proteins are full of it on the surface. Use those six for the headline, and report the wider
> set alongside so we can see both.

**What counts as still folding into the backbone?**

> TM-score 0.8 is the right line, and I prefer it to RMSD — our designs vary in length and RMSD
> punishes a wagging tail more than it should. A design that fails the fold earns nothing.

**Which backbones should it be judged on?**

> Single chain, 60 to 300 residues, compact single domains. Design the whole chain.

**How many sequences per backbone?**

> Eight. That's what we'd actually order.

**Budget?**

> About 12 GPU-hours, on the free queue.

### Then the agent pushed back

Before freezing, it checked whether the metric could be gamed, and found that excluding alanine
left a hole: a model could swap every exposed F for an A and satisfy the metric without changing
anything real.

> You've caught me contradicting myself. My worry was that the model would please the metric by
> swapping greasy residues for alanines — but if alanine isn't counted, that swap is exactly the
> free win I was afraid of. Counting it closes the hole. Switch the headline to AFILMVWY, and
> keep my six beside it.

Approved, and frozen.

## What you get

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

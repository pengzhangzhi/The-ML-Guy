# Pinning down the problem

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

## Then the agent pushed back

Before freezing, it checked whether the metric could be gamed, and found that excluding alanine
left a hole: a model could swap every exposed F for an A and satisfy the metric without changing
anything real.

> You've caught me contradicting myself. My worry was that the model would please the metric by
> swapping greasy residues for alanines — but if alanine isn't counted, that swap is exactly the
> free win I was afraid of. Counting it closes the hole. Switch the headline to AFILMVWY, and
> keep my six beside it.

Approved, and frozen.

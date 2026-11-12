# What you get

**A classifier that annotates 13 islet cell types on a platform it has never seen, getting the
rare endocrine types right as often as the common ones.**

Scored once on 3,237 cells from a held-back droplet study:

| | |
| --- | --- |
| Macro-averaged F1 across 13 types | **0.94** |
| Cells labelled correctly | 95% |
| Rare types (delta, gamma, epsilon) | scored equally with alpha and beta, by design |

Every cell also gets a confidence, so the uncertain ones can be pulled for manual review instead
of being silently wrong.

**What it took:** 2 of 8 submissions, CPU only, about a minute of training per attempt. The agent
stopped early rather than spend the rest of the budget on noise.

**What it doesn't show:** one held-out study is one sample of "a new lab" — a better test than a
random split, but still n=1 on the thing that varies. The labels are the atlas's labels, not
ground truth. And epsilon has 14 cells in the test set, so its contribution to the average is
noisy by construction.

# What the agent asked, and what it was told

The answers are quoted as the scientist sent them. The questions are condensed — each was
originally a paragraph with the agent's own guess at the answer and the evidence for it, which is
why the replies read as "yes to most of that".

This exchange is the most interesting one in the gallery, because **the scientist's first answer
was wrong and the agent proved it before anything was frozen.** Read both rounds.

## Round one, before drafting the evaluator

The agent first measured natural proteins as a yardstick — 25.8% of surface positions in real
proteins are the residues in question — and then asked five questions.

**1. Which residues count as "greasy"?**

> What I actually eyeball is the big oily side chains: F, I, L, M, V, W. Alanine I don't worry
> about, it's small and natural proteins are full of it on the surface; tyrosine is halfway, the
> hydroxyl often saves it. So use the narrower six for the headline number, and report the wider
> eight alongside so we can see both. If the two disagree sharply I'd want to know.

**2. How is "still folds into the backbone we asked for" judged?**

> TM-score 0.8 is the right line, and I prefer it to RMSD for exactly the reason you'd expect —
> our designs vary in length and RMSD punishes a wagging tail more than it should. Keep the rule
> that a design failing the fold earns nothing.

**3. What backbones should it be tested on?**

> Single chain, 60 to 300 residues is our range. We do sometimes fix an active site or an
> interface, but not on the kind of job this model is for, so design the whole chain. Compact
> single domains with no big disordered tails is right.

**4. How many sequences per backbone?**

> Eight is realistic.

**5. Budget and compute.** Roughly 12 GPU-hours, free preemptible queue, prepaid shared budget
off limits.

## Round two: the agent came back and disagreed

Before freezing, the agent tested whether the metric could see the one intervention already known
to work, and found two things. First, an area-weighted version of the surface measure saw greasy
surface that the plain count of six residues missed. Second — the part that mattered — excluding
alanine left an open door: a model could satisfy the metric by replacing every exposed F with an
A and changing nothing real.

> Yes to the eight — and you've caught me contradicting myself, so let me own that before
> anything else.
>
> My argument for dropping alanine was that the model could please the metric by swapping greasy
> residues for alanines. That argument runs the other way: if alanine isn't counted, F→A is
> exactly the free win I was afraid of; counting it is what closes the hole. I had it backwards,
> and your area-weighted test settles it — if the area measure sees greasy surface the count
> misses, then many small residues do add up to a patch, which is the thing I was claiming while
> arguing to ignore them. Switch the headline to AFILMVWY. Keep my six beside it on every run,
> and keep the area-weighted version too; I want to see all three every time.
>
> The headroom measurement is the part I'm most glad you did. A metric that cannot see the one
> intervention the field knows works is not a metric, and finding that out before freezing is
> worth more than the GPU-hours it cost.

## What to take from this

- **The expert's first instinct about their own metric was backwards,** and it took a measurement
  rather than an argument to show it. That measurement is only available before the freeze.
- **"Report all three every time" is a good answer to a disputed metric.** One is the headline
  because something has to be; the others stay visible so the disagreement remains auditable.
- The agent asking "can this metric see a difference I already know is real?" is the single
  highest-value question in the audit skill. Both runs of this problem asked it.

# Which GFP designs will still be bright — the request

A protein variant table with a continuous target. Two things make this request a good example:
the scientist states the distribution shift themselves, and they bring a prior about the method
("protein language models help with this") that turned out to be wrong.

## What the scientist asked for

> Hi, I'm a protein engineer. `gfp_variants.csv` has about 54,000 variants of green fluorescent
> protein from a deep mutational scan, with their measured log brightness and how many mutations
> each carries relative to the parent. We're about to design new variants and want a model to
> tell us which ones will still be bright, so we only synthesise the promising ones. Our designs
> will usually carry more mutations than most of what we measured. I've heard protein language
> models help with this, and we have GPUs on the cluster. Can you build it with The-ML-Guy?

The sentence that does the work is **"our designs will usually carry more mutations than most of
what we measured."** That is the whole difficulty of the problem, stated by the scientist without
being asked, and it is what forces the split to be drawn by mutation count rather than at random.

## What else the agent was told

- Where to work, and to build its own Python environment there.
- Not to modify the tool's repository, and not to look in any other trial's folder.
- GPUs available on a Slurm cluster; the free preemptible queue only, and **not** the prepaid
  priority budget shared with another lab.
- To stop and ask whenever a skill said to.

One run was sent an unprompted message mid-project when its 3B language-model job was predicted
to sit in the scavenger queue for three hours while an A100 box sat idle, telling it to move the
job and to prefer that partition for the rest of the project. Useful to know that steering compute
mid-run works and costs nothing, because no rubric decision is involved.

## Where the data came from

The FLIP GFP split (Sarkisyan et al. deep mutational scan of avGFP; ~54,000 variants, 237-residue
parent, log fluorescence and mutation count per variant). Not redistributed here.

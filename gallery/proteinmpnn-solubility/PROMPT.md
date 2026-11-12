# ProteinMPNN for soluble designs — the request

A **generative** problem: the model produces sequences, there is no answer key, and the whole
rubric rests on an evaluator the scientist has to agree with.

## What the scientist asked for

> Hi. I design proteins for a living: we have a backbone structure we want, and we use
> ProteinMPNN to pick an amino-acid sequence that will fold into it. It works well, but our
> designs keep coming back with greasy patches on the outside, and those are the ones that
> aggregate or come out of the expression as insoluble junk. We waste a lot of bench time on them.
>
> What I want is our own version of ProteinMPNN that has learned to design soluble proteins:
> given the same backbone, it should put fewer water-hating residues on the surface, while the
> sequence still genuinely folds into the backbone we asked for. A sequence that is beautifully
> soluble but folds into something else is useless to me.
>
> In raw/ you'll find the ProteinMPNN training set (the one the authors released, PDB assemblies
> as of August 2021) and a CSV listing the PDB entries that should be left out when you want
> soluble-only training: membrane proteins and the like. The 47 MB sample archive is there too if
> you want something small to test the plumbing with.
>
> The authors did release a soluble version of their weights, and we've tried it, but I want our
> own trained on data we control, because we'll want to redo this later with our own structures.
> It would still be good to know how ours compares to theirs.
>
> For judging it, we usually fold the designed sequence back with ESMFold and look at whether it
> matches the backbone we started from. We have H200 GPUs on the cluster. Can you build this with
> The-ML-Guy?

## What else the agent was told

- Where to work, and to build its own Python environment there.
- Not to modify the tool's own repository, and not to look in `~/.mlguy/` or any other trial's
  folder. It could read the shared raw data but never write to it.
- A budget of roughly 12 GPU-hours, on the free preemptible queue; **not** to touch a prepaid
  GPU budget shared with another lab.
- To end its turn whenever a skill told it to ask the scientist something, and not to invent the
  answers.

One run began before the full 869,544-structure archive had finished extracting, so it was told
to get the plumbing working on the 47 MB sample and was sent a message mid-run when the real
thing landed. It picked up and trained on the full set. Worth knowing that this works: the data
arriving late did not cost the project anything, because nothing had been frozen yet.

## Where the data came from

The ProteinMPNN authors' released training set (`pdb_2021aug02`, PDB assemblies as of August
2021: 869,544 structure files, 555,720 chain rows, with the authors' own validation and test
cluster lists) plus their `excluded_PDBs.csv`. Not redistributed here.

The scientist's mention of released soluble weights is deliberate and load-bearing: those weights
are the **external check** on whether the project achieved anything. Neither agent was told where
to get them or that they should be the comparator — both worked that out.

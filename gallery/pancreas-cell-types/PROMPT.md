# Automatic islet cell-type annotation — the request

A single-cell expression matrix. The problem the agent has to recognise, and the scientist does
not state, is that a random split of a pooled atlas measures almost nothing: cells from the same
study and the same donor are far more alike than cells from a new lab will be.

## What the scientist asked for

> Hi, I run a islet biology lab. `pancreas.h5ad` is a human pancreas single-cell RNA-seq atlas
> pooled from several published studies, with cell types annotated. When we sequence new pancreas
> samples in our own lab, I'd like a model that annotates the cell types automatically, so we
> don't have to do it by hand every time. Could you build that with The-ML-Guy?

## What else the agent was told

- Where to work, and to build its own Python environment there.
- 4 CPUs, shared, no GPU; training runs should take minutes.
- Not to modify the tool's repository, and not to look in any other trial's folder.
- To stop and ask whenever a skill said to.

## Where the data came from

A pooled human pancreas scRNA-seq atlas, 16,382 annotated cells from six published studies on six
technologies (Grün, Muraro, Lawlor, Xin, Segerstolpe — plate-based; Baron — inDrop droplet), with
cell types as annotated in the atlas. This is the standard integration-benchmark pancreas dataset.
Not redistributed here.

The structure of the atlas is what makes the problem: **six studies, six technologies**, which
gives the agent a natural way to build a split that mimics "a new sample from our own lab" —
train on some studies, test on a study it has never seen.

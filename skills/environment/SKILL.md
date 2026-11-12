---
name: environment
description: Find out what the machine you are working on offers (CPUs, memory, GPUs, scheduler, storage, Python environments) and write it down once, so every later step can rely on it. Use at the start of a The-ML-Guy project, when a job fails or sits pending for reasons that look like the machine, or when PROJECT_MEMORY.md has no "Environment" section.
---

# Learning the machine

What is worth trying depends on the compute, so learn it before planning any modelling. This is
yours to discover, not the user's to explain. Ask them only what cannot be found out: which
account to charge, what they are willing to spend.

## Read before you probe

Someone may have written it down already. Check, in this order: the project's
`PROJECT_MEMORY.md`, the machine's notes (`CLAUDE.md`, `AGENTS.md`), and your own memory. Trust
what you find for where things are; re-check anything that changes, such as limits and what is
free.

## What to find out

- **Compute:** CPUs, memory, GPUs and their type and memory (`nproc`, `free -g`, `nvidia-smi`).
- **Scheduler:** whether there is one. With Slurm: `sinfo`, `sacctmgr show assoc user=$USER`,
  `squeue`. Which partitions and accounts you may use, their limits, whether jobs can be
  preempted, and whether any allocation is a budget that runs out.
- **Storage:** where large files, environments and caches should live, and which locations are
  small or slow (`df -h`, quotas). A home directory is often the wrong place.
- **Software:** which Python and package manager to use, whether an environment already exists
  for this project, and whether the machine can reach the internet to download weights and data.

**What a queue can run is not what it will run now.** The partition with the right GPUs may be
full, and a submission budget spent waiting is worse than a smaller GPU that starts immediately.
Look at what is idle (`sinfo` by state, `squeue -p <partition>`) before choosing, and look again
when a job sits pending: a scheduler's own start estimate is usually pessimistic and is no
substitute for seeing which nodes are free.

## Write it down

Put what you found in `PROJECT_MEMORY.md` under "Environment", as commands someone can copy:
the interpreter or environment to activate, the exact scheduler flags that work, where scratch
files go. Save it to your own memory too, if you have one, so the next project on this machine
starts from it.

Every later step reads that section instead of rediscovering it: the audit, which starts with
none of what you know; the research loop, which has to run `mlguy submit` where the model can
run; and whoever picks the project up next.

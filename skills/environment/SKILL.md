---
name: environment
description: Learn the user's development environment for ML work — compute and GPUs, the job scheduler, storage, where code and environments should live, and how to build a working Python environment — and write it down once per machine so every later step and project starts from it. Use at the start of a The-ML-Guy project, when the machine notes are missing or stale, or when a job fails or waits for reasons that look like the machine.
---

# Learning the machine

Every user's machine is different: a laptop, a workstation with one GPU, a cluster with a
scheduler, accounts, quotas and rules. What is worth trying depends on it, and every hour the
research loop spends rediscovering it is an hour not spent on the model. Learn it once, write it
down, and let everything after read it.

This is yours to discover, not the user's to explain. Ask them only what cannot be found out.

## Read before you probe

Look for notes first, in this order:

1. `${MLGUY_HOME:-~/.mlguy}/environment.md`, written by this skill on an earlier project.
2. The machine's agent notes (`~/.claude/CLAUDE.md`, `AGENTS.md`) and your own memory.

Trust them for where things are and how they are done. Re-check what changes: limits, quotas,
what is free right now.

## What to find out

- **Compute:** CPUs, memory, GPUs, their type and memory (`nproc`, `free -g`, `nvidia-smi`). On a
  cluster the machine you are on is usually a login node with no GPU; what matters is what a job
  gets.
- **Scheduler:** with Slurm, `sinfo`, `sacctmgr show assoc user=$USER`, `squeue`. Which partitions
  and accounts you may use, the flags each needs, time limits, whether jobs can be preempted, and
  whether any allocation is a budget that runs out or is shared with other people.
- **Storage:** where code, environments, data, model weights and caches should live, and which
  places are small, slow or purged (`df -h`, `quota`). A nearly full home directory is common, and
  caches written there by default (`~/.cache`) are the usual way it fills.
- **Python:** which interpreter and package manager to use, whether a usable environment already
  exists, and the CUDA version the driver supports.
- **Network:** whether compute nodes can download packages, weights and data, or only the login
  node can.

**What a queue can run is not what it will run now.** The partition with the right GPUs may be
full, and a submission budget spent waiting is worse than a smaller GPU that starts immediately.
Look at what is idle (`sinfo` by state, `squeue -p <partition>`) before choosing, and look again
when a job sits pending: a scheduler's own start estimate is usually pessimistic.

## Ask only what cannot be found out

Which account to charge, what they are willing to spend, and any rule that is theirs rather than
the machine's: an allocation they share and must not drain, a partition they would rather keep
free. One short message, after you have looked, so it can name what you found.

## Prove it works

Before writing anything down, run the smallest real thing end to end: build the project's Python
environment where it belongs, with caches pointed away from small disks (`UV_CACHE_DIR`,
`PIP_CACHE_DIR`, `HF_HOME`, `TORCH_HOME`), then submit a one-minute job that imports the framework
the project will use and confirms it sees the GPU. A note that was never run is a guess.

## Write it down

**Once per machine,** in `${MLGUY_HOME:-~/.mlguy}/environment.md`, so the next project on this
machine skips most of this skill. Write it as commands someone can copy:

- the job flags that work, per partition, and what each is for
- where code, environments, data and caches go, and the variables that send them there
- how to build an environment, and how to activate it inside a job
- the user's rules, in their words

**Per project,** an "Environment" section in `PROJECT_MEMORY.md`: the environment this project
uses, how to activate it, and a pointer to the machine notes. The audit, the research loop and
whoever picks the project up next read it instead of rediscovering the machine.

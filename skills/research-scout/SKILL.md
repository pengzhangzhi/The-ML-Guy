---
name: research-scout
description: Find and distill task-relevant scientific papers and open-source ML implementations into actionable, evidence-backed RESEARCH_CONTEXT.md notes. Use during project setup before evaluation is frozen, before substantial modeling, and again during autoresearch when new knowledge would help.
---

# Research Scout

Help The ML Guy make better research decisions by learning from existing scientific work. **Do not write a generic literature survey.** Research the scientist's *specific task* and produce implementation-ready guidance that the modeling agent can choose to use.

This is a knowledge-gathering skill, not a new agent framework. Research Scout does not train or reproduce external models, change modeling code, or choose the final approach.

## When to use

- **During project setup:** Research established task definitions, datasets, metrics, validation splits, baselines, and evaluation pitfalls **before** the scientist approves and the evaluator is frozen.
- **Before substantial modeling:** Find relevant architectures, training approaches, implementations, and lessons from closely related tasks.
- **During autoresearch:** Revisit a specific knowledge gap, failed hypothesis, unfamiliar method, or technical obstacle when further research would inform the next decision. Do not repeat a broad search without a reason.

Keep the initial investigation proportional to the problem. Small, familiar tasks may need only a brief scan.

## Workflow: scope → ground → distill → deliver

1. **Scope the capability gaps.** Read the scientist's goal, available data and data profile, task brief/evaluator if present, compute constraints, and any existing `RESEARCH_CONTEXT.md`. Identify the questions that matter to the *next decision*: scientific assumptions, evaluation, modeling, or implementation. Do not search the user's wording alone.
2. **Discover then investigate adaptively.** Search relevant papers, benchmark descriptions, GitHub repositories, documentation, and credible technical discussions. Start with directly related work; explore analogous methods from other fields where they might transfer. Scan abstracts/READMEs first, then read the few high-value papers and code paths in depth. Prefer methods with credible evidence and usable implementations, but do not exclude promising new ideas.
3. **Ground each useful finding.** Check original paper sections and, where important, actual source/config/test files. Record links, relevant versions or commits when available, and exact files/functions when known. Separate **reported paper results**, **observed repository details**, and **our untested hypotheses**. Never imply that code was run or results reproduced if they were not.
4. **Distill for the current task.** For each significant finding, explain: **when it applies → what to consider → why → how to implement/adapt → evidence → limitations**. Include concrete code references, expected dependencies, and relevant failure modes if known. Prefer a few actionable findings over a long source list.
5. **Deliver incrementally.** Create or update `RESEARCH_CONTEXT.md` in the project root. Read it before editing; preserve valid prior findings, refine weak conclusions, remove duplicates, and explicitly correct outdated or contradicted claims. Summarize the highest-value recommendations briefly for the main agent. If researching in a subagent, return findings to the parent unless one writer has been designated for the file.
6. **Stop when useful.** Stop once the next evaluation or modeling decision is sufficiently informed and further searching offers little expected value. Resume with a targeted question when useful; record unresolved gaps rather than inventing certainty.

## RESEARCH_CONTEXT.md

Keep this file concise and navigable; add detail only where it changes a decision.

```markdown
# Research Context
## Task and knowledge gaps
- Goal, constraints, and unresolved questions.

## Actionable findings
### [Finding]
- Use when:
- Consider / how to adapt:
- Why it may help:
- Evidence: [paper link/section], [repo link/file/commit]
- Status: reported finding | observed code | hypothesis
- Limits / uncertainties:

## Evaluation guidance
- Established metrics/splits/baselines; applicability and caveats.
- Mark suggestions requiring user approval before task freeze.

## Promising next investigations
- Specific approaches, implementation leads, or hypotheses to test.

## Open questions
- Missing evidence or research that may become useful later.
```

Avoid duplicating large paper summaries or copied source code. Link to the originals. If little relevant literature exists, state that clearly and provide the best-supported alternatives.

## Boundaries

- **The ML Guy chooses and implements.** Scout provides knowledge and references, not code changes, experiments, model selection rules, or a mandatory experiment plan.
- **Evaluation remains the scientist's contract.** Before approval, inform the evaluation proposal. After `task/` is frozen, do not edit it or suggest silently changing metrics/splits; surface concerns separately.
- **Protect evaluation integrity and privacy.** Do not search held-out labels, reveal private data to external services, or tailor modeling advice to the hidden test set.
- **Be source-faithful.** Attribute claims precisely, note missing access, conflicting findings and source age, and distinguish an author's benchmark from evidence that an approach will work here.
- **Stay lightweight.** No generated skill graphs, package installation, mandatory execution checks, external model reproduction, or new infrastructure. Existing agent tools and optional subagents are enough.

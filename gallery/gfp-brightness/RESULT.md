# What you get

**Give it a few thousand designs carrying 6 to 12 mutations, synthesise the 100 it ranks
highest, and about 93 of them will be bright.**

Scored once on 4,655 variants the model never saw:

| How you pick 100 to synthesise | How many are bright |
| --- | ---: |
| At random | 5 or 6 |
| Fewest mutations first (current practice) | 14 |
| **This model** | **93** |

About a 7× improvement on current practice — and "current practice" is a measured number, not an
assumed one, because the agent measured it during briefing.

**It also answered the question you asked in passing.** Protein language models did not help
here: ESM-2's zero-shot score ranks at chance on this problem (AUROC 0.54), and adding it as
features changed nothing. The agent spent one submission establishing that so the report could
say it with a number. The model that shipped is a ridge-plus-logistic blend on substitution
indicators — no language model, no GPU needed to run it.

**What it took:** 5 of 12 submissions. The agent stopped early.


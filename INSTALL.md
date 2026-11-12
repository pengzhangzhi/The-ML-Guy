# Installing The-ML-Guy

Written so that a coding agent can follow it. If you are a person, the same steps work; paste
them into Claude Code or Codex and say "install this tool" and it will do them for you.

## 1. Install the command line tool

Needs Python 3.11 or newer.

```bash
git clone https://github.com/pengzhangzhi/The-ML-Guy.git
cd The-ML-Guy
pip install -e .
mlguy --help          # should list: new, init, validate, submit, finish, status, ...
```

With `uv`, which is faster and keeps the environment separate:

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
```

The core install is only what running and recording a project needs (numpy, pandas).
Each project's evaluator, model and page bring their own libraries (scikit-learn, scanpy,
lightgbm, torch, gradio...), installed by the agent when that project needs them.

## 2. Install the skills

The skills tell the agent how to use the tool. Copy them where your agent looks for skills.

**Claude Code** — for every project on this machine:

```bash
mkdir -p ~/.claude/skills
cp -r skills/* ~/.claude/skills/
```

Or, for one project only, copy them into `.claude/skills/` inside that project.

**Codex** — Codex reads `AGENTS.md` rather than a skills folder. Point it at the skills from
the `AGENTS.md` of the project you are working in:

```markdown
## Building ML models

This machine has The-ML-Guy installed. When asked to build a prediction model from data,
read and follow /absolute/path/to/The-ML-Guy/skills/build-project/SKILL.md, then
skills/research/SKILL.md, then skills/deploy/SKILL.md. When build-project asks for an
audit, start a separate Codex session (`codex exec`) that follows skills/audit/SKILL.md.
Run `mlguy --help` for the commands.
```

Both have been run end to end on real projects.

## 3. Check it works

```bash
mlguy new smoke-test --name smoke-test && cd smoke-test && mlguy status
```

It should list what was scaffolded, then report the project with its metric still `TODO` and
nothing recorded yet. Delete the folder afterwards; a real project starts from a conversation,
not from a template you fill in alone.

## Where things are kept

Test labels, the hash of the frozen task, and the final test score live in
`~/.mlguy/projects/<project-name>/`, outside the project folder, so that training code cannot
read them by accident. Set `MLGUY_HOME` before `mlguy init` to put them somewhere else; the
project remembers it, so later commands need not set it. Once `mlguy finish` has scored the test set, it
copies the labels into the project's `final/` folder, since there is nothing left to protect.

## What this does not do

There is no sandbox. The agent runs with your permissions, like any coding-agent session.
Scores are protected against accidents, not against an agent that deliberately goes looking
for the test labels.

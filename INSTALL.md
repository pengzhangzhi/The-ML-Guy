# Installing The-ML-Guy

Written so that a coding agent can follow it. If you are a person, the same steps work; paste
them into Claude Code or Codex and say "install this tool" and it will do them for you.

## 1. Install the command line tool

Needs Python 3.11 or newer.

```bash
git clone https://github.com/pengzhangzhi/The-ML-Guy.git
cd The-ML-Guy
pip install -e ".[dashboard]"
mlguy --help          # should list: new, init, validate, submit, finish, status, ...
```

With `uv`, which is faster and keeps the environment separate:

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dashboard]"
```

The `dashboard` extra is optional. There is no page to install: the agent builds one per
project with whatever library suits it. Everything a generated project needs to train is in the core install.

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
skills/research/SKILL.md, then skills/deploy/SKILL.md.
Run `mlguy --help` for the commands.
```

Claude Code is the tested path. Codex should work, but has not been run end to end yet.

## 3. Check it works

```bash
cd examples/amylase
python make_task.py && mlguy init && python src/train.py && mlguy validate
```

The last command should print a validation score of about 0.68 and say the loop is open.

## Where things are kept

Test labels, the hash of the frozen task, and the final test score live in
`~/.mlguy/projects/<project-name>/`, outside the project folder, so that training code cannot
read them by accident. Set `MLGUY_HOME` to put them somewhere else.

## What this does not do

There is no sandbox. The agent runs with your permissions, like any coding-agent session.
Scores are protected against accidents, not against an agent that deliberately goes looking
for the test labels.

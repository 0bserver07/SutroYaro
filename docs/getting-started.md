# Getting Started

How to go from cloning the repo to running your first experiment with a coding agent.

## 1. Clone and verify

```bash
git clone https://github.com/cybertronai/SutroYaro.git
cd SutroYaro

# Option A: Nix (recommended - includes all deps)
nix develop
python3 checks/env_check.py
python3 checks/baseline_check.py

# Option B: pip (fallback)
export PYTHONPATH=$PWD/src:$PYTHONPATH
pip install numpy
python3 checks/env_check.py
```

Both checks must pass. If they don't, fix the issue before continuing.

See [CONTRIBUTING.md](https://github.com/cybertronai/SutroYaro/blob/main/CONTRIBUTING.md) for full setup instructions including how to install nix.

## 2. Open in your agent

Open the repo in whatever coding agent you use. The agent will read the project context automatically:

| Agent | What it reads | Command |
|-------|--------------|---------|
| Claude Code | CLAUDE.md | `claude` in the repo directory |
| Gemini CLI | GEMINI.md (if present) or project files | `gemini` in the repo directory |
| Antigravity | Project files via VS Code | Open folder in Antigravity |
| Cursor | .cursorrules (if present) or CLAUDE.md | Open folder in Cursor |

If your agent doesn't auto-read CLAUDE.md, tell it: "Read CLAUDE.md, DISCOVERIES.md, and TODO.md."

## 3. What the agent sees

The agent picks up context from these files in order:

1. **CLAUDE.md** -- project context, current best methods, constraints, working style
2. **DISCOVERIES.md** -- what's proven so far, what failed, open questions
3. **TODO.md** -- hypothesis queue with unchecked items
4. **AGENT.md** -- the experiment loop protocol (if running autonomous)
5. **LAB.md** -- experiment rules (metric isolation, one hypothesis per experiment)

The agent should read DISCOVERIES.md before doing anything so it doesn't repeat known results.

## 4. Pick a task

Three options:

**Run an existing experiment.** Pick any method from the [survey](https://cybertronai.github.io/SutroYaro/research/survey/) and verify the numbers on your machine.

```
"Run the GF(2) experiment and verify it matches the survey results"
```

**Try an open hypothesis.** TODO.md has unchecked items with paper references. Tell the agent to pick one.

```
"Read TODO.md and try the next unchecked hypothesis"
```

**Add a new challenge.** Follow the [adding-a-challenge guide](research/adding-a-challenge.md) to add a new task to the harness.

```
"Read docs/research/adding-a-challenge.md and add a sparse-majority challenge"
```

## 5. Run experiments

The agent uses the harness. If using nix, PYTHONPATH is set automatically:

```bash
# Sparse parity (default)
python3 src/harness.py --method gf2 --n_bits 20 --k_sparse 3

# Sparse sum
python3 src/harness.py --challenge sparse-sum --method sgd

# All 14 experiments in 0.28 seconds
python3 bin/reproduce-all
```

Without nix, prefix with `PYTHONPATH=src`.

## 6. Record results

Every experiment produces:
- Code in `src/sparse_parity/experiments/`
- Results JSON in `results/`
- Findings doc in `findings/` (use `findings/_experiment_template.md`)
- Update to DISCOVERIES.md if it answers an open question

## 7. Submit your work

See [branch workflow](branch-workflow.md) for how to create a branch and submit a PR.

## Existing docs

| Doc | What it covers |
|-----|---------------|
| [CONTRIBUTING.md](https://github.com/cybertronai/SutroYaro/blob/main/CONTRIBUTING.md) | Three levels of contribution effort, PR process |
| [Agent CLI Guide](tooling/agent-cli-guide.md) | Setup for Claude Code, Gemini CLI, Codex, Antigravity |
| [Claude Code Setup](tooling/claude-code-setup.md) | How CLAUDE.md and LAB.md work |
| [Adding a Challenge](research/adding-a-challenge.md) | Step-by-step guide to add new tasks |
| [Sync Runbook](tooling/sync-runbook.md) | How to sync Telegram, Google Docs, GitHub |

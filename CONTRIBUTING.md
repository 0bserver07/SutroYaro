# Contributing

Anyone in the Sutro Group (or interested outsiders) can contribute experiments, findings, or raw results.

## The short version

1. Fork the repo
2. Run an experiment on sparse parity (with any tool: Claude Code, Replit, Gemini, plain Python, pencil and paper)
3. Submit a PR with your findings

We accept contributions at three levels of formality. Pick the one that fits your workflow.

## Level 1: Drop raw results (lowest effort)

Put whatever you have in `contributions/`. No format required.

```
contributions/
  germain-depth1-results.md    # copy-paste from your agent's output
  michael-claude-log.txt       # raw Claude conversation
  notes-from-monday.md         # meeting notes, ideas, observations
```

Someone (usually Claude Code) will reformat it later. The point is to get the information into the repo, not to make it pretty.

## Level 2: Write a findings doc (medium effort)

Copy the template and fill it in:

```bash
cp findings/_template.md findings/exp_your_name.md
```

The template asks for: hypothesis, config, results table, analysis, open questions. See any file in `findings/` for examples. This is the format that feeds into the [survey](https://cybertronai.github.io/SutroYaro/research/survey/).

## Level 3: Full experiment with code (highest effort)

Follow the lab protocol:

```bash
# 1. Read what's known
cat DISCOVERIES.md

# 2. Copy the code template
cp src/sparse_parity/experiments/_template.py src/sparse_parity/experiments/exp_yours.py

# 3. Edit, run, measure
PYTHONPATH=src python3 src/sparse_parity/experiments/exp_yours.py

# 4. Write findings
cp findings/_template.md findings/exp_yours.md

# 5. Submit PR with all three: experiment code, results JSON, findings doc
```

## Setup

### Option A: Nix (recommended)

One command gets you everything: Python, numpy, bun, pandoc, mkdocs.

```bash
# 1. Install nix (if you don't have it)
curl --proto '=https' --tlsv1.2 -sSf https://install.determinate.systems/nix | sh -s -- install

# 2. Clone and enter dev shell
git clone https://github.com/cybertronai/SutroYaro.git
cd SutroYaro
nix develop

# 3. Verify it works
python3 -m sparse_parity.fast
# Should print: 100% accuracy in ~0.12s

# 4. Run environment check
python3 checks/env_check.py
# Should print: All checks passed.
```

The dev shell sets `PYTHONPATH=src` automatically, so imports work without manual setup.

**What's included:**
- Python 3 + numpy (core experiments)
- bun (Telegram sync)
- pandoc (Google Docs sync)
- mkdocs-material + plugins (docs site)

**Direnv (optional):** If you use direnv, `direnv allow` will auto-load the dev shell when you cd into the repo.

### Option B: pip/uv (fallback)

If you don't want to install nix:

```bash
git clone https://github.com/cybertronai/SutroYaro.git
cd SutroYaro

# Core deps only (experiments)
pip install numpy

# Or with uv
uv pip install numpy

# Set PYTHONPATH and verify
export PYTHONPATH=$PWD/src:$PYTHONPATH
python3 -m sparse_parity.fast
```

Optional deps for sync/docs:
```bash
pip install mkdocs-material mkdocs-mermaid2-plugin pymdown-extensions
# pandoc: brew install pandoc (macOS) or apt install pandoc (Linux)
# bun: curl -fsSL https://bun.sh/install | bash
```

## What to work on

Open questions live in [DISCOVERIES.md](DISCOVERIES.md) under "Open Questions." The [task tracker](docs/tasks/INDEX.md) has current priorities. Some starting points:

- **Reproduce a result**: pick any experiment from the [survey](https://cybertronai.github.io/SutroYaro/research/survey/) and verify the numbers
- **Try a new algorithm**: check [proposed-approaches.md](docs/research/proposed-approaches.md) for untested ideas
- **Improve the metric**: we just added DMC (Data Movement Complexity) alongside ARD. Both could use testing on more configs.
- **Scale testing**: does your method work at n=50/k=5? n=100/k=10?

## Rules

1. **Read DISCOVERIES.md before experimenting** so you don't repeat what's known
2. **One hypothesis per experiment** so results are interpretable
3. **Always compare against a baseline** so numbers have context
4. **Don't modify measurement code** (tracker.py, cache_tracker.py, data.py, config.py). If you need to change the metric, that's a separate PR.
5. **Record negative results** because knowing what doesn't work prevents others from trying it

## PR process

- Fork and branch
- Push your branch, open a PR against `main`
- Describe what you tried and what happened
- Someone (human or AI) will review and merge

No CI gates. No test coverage requirements. The only hard rule is: don't break existing experiments.

## Questions

Ask in the [Telegram group](https://t.me/sutro_group) or open a GitHub issue.

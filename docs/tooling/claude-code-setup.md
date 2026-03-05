# Claude Code Setup

How this repo is configured for Claude Code research automation.

## CLAUDE.md

The file `CLAUDE.md` at the repo root is read by Claude Code at the start of every session. It contains:

- **Project context** — What the Sutro Group is, what we're researching
- **Core concepts** — Sparse parity, ARD, grokking, CacheTracker
- **Current best config** — The hyperparameters that work (n=20, k=3, hidden=200, lr=0.1)
- **Findings** — Summary of what's been proven so far
- **Working style** — Iteration time < 2s, one change at a time, priority ordering

This means every new Claude Code session already knows the project state without re-explaining.

### Key Rules in CLAUDE.md

```
- Iteration time must stay under 2 seconds
- Change one thing at a time
- Priority: correctness > wall-clock time > energy usage
- One hypothesis per experiment, always compare against baseline
- Record everything — failed hypotheses are findings too
```

## LAB.md

The experiment protocol. Claude Code reads this before running any experiment. It enforces:

- Experiment template (hypothesis, method, results, findings)
- Lifecycle: proposed → running → completed/failed
- Baseline comparison for every experiment
- Commit after every experiment

## Memory

Claude Code has a persistent memory directory at `~/.claude/projects/.../memory/`. It stores:

- `MEMORY.md` — Key facts loaded every session (hard rules, project conventions)
- Topic-specific files (e.g., `mkdocs-notes.md`) linked from MEMORY.md

This is how Claude Code remembers things like "dropdown tabs are not supported in Material for MkDocs" across sessions.

## Settings Worth Trying

### Permission Modes

| Mode | When to use |
|------|------------|
| Default | Normal work — Claude asks before running commands |
| `--dangerously-skip-permissions` | Batch automation where you trust the workflow |
| Plan mode | Complex tasks — Claude writes a plan, you approve, then it executes |

### Parallel Agents

Claude Code can spawn sub-agents for independent tasks. We used this to run multiple experiments simultaneously:

```
# Claude Code spawns agents like:
Agent(subagent_type="general-purpose", isolation="worktree")
```

Each agent gets its own git worktree, runs independently, and reports back.

### Hooks

Shell commands that run in response to Claude Code events. Potential uses:

- Auto-run `python3 src/sync_google_docs.py` after editing docs config
- Run experiment validation after writing a new experiment file
- Lint markdown files before committing

### Model Selection

| Model | Use case |
|-------|---------|
| Opus | Complex research, experiment design, multi-file changes |
| Sonnet | Fast iteration, simple edits, running scripts |
| Haiku | Quick searches, file reads, simple questions |

## MCP Servers

Model Context Protocol servers extend Claude Code with new tools. Relevant ones:

- **File system MCP** — Direct file access without bash
- **Google Docs MCP** — Could replace the sync script with live access (not yet set up)
- **Browser MCP** — For fetching web content, papers, etc.

## Repo Structure for Claude Code

```
CLAUDE.md          — Project context (read first every session)
LAB.md             — Experiment protocol
DISCOVERIES.md     — Proven findings (read before every experiment)
TODO.md            — Open research tasks
.claude/
  skills/          — Project-level skills (auto-detected by Claude Code)
    anti-slop-guide/SKILL.md
src/               — Experiment scripts
  fast.py          — Numpy solver (< 2s iteration)
  sync_google_docs.py — Google Docs pull script
docs/              — MkDocs site
  findings/        — One markdown file per experiment
  google-docs/     — Pulled Google Docs
  tooling/         — This section
results/           — Raw experiment output
```

Anyone cloning this repo gets the skills automatically. Claude Code detects `.claude/skills/` on session start.

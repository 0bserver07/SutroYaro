# .claude/ directory

Claude Code configuration for this project. Other coding agents (Gemini CLI, Codex, Antigravity) do not use these files -- they read CLAUDE.md at the repo root instead.

## What's here

### hooks/

Node.js scripts that run automatically during Claude Code sessions.

| File | Event | What it does |
|------|-------|-------------|
| `session-start.cjs` | SessionStart | Shows git status, open issues, last experiment, sync dates |
| `security-guard.cjs` | PreToolUse | Blocks edits to locked measurement code (tracker.py, cache_tracker.py, data.py, config.py) and destructive commands (rm -rf, git reset --hard). Two tiers: deny (blocks) and confirm (asks user for git push --force). |
| `session-end.cjs` | SessionEnd | Shows uncommitted changes, recent commits, stale task warnings |
| `hook-common.cjs` | (library) | Shared utilities used by the other hooks |

### rules/

Markdown files loaded as constraints for the agent.

| File | What it enforces |
|------|-----------------|
| `experiment-reproducibility.md` | Seeds, config logging, environment recording, baseline comparison |
| `agent-coordination.md` | Parallel dispatch criteria, file ownership table, review gates |

### skills/

Reusable workflows the agent can invoke. Each is a directory with a `SKILL.md` and optional `examples/`.

| Skill | When to use |
|-------|------------|
| `sutro-sync` | Sync Google Docs, Telegram, GitHub |
| `sutro-context` | Load current project state before research |
| `anti-slop-guide` | Review prose for AI writing patterns |
| `run-experiment` | Run an experiment following two-phase protocol |
| `weekly-catchup` | Generate weekly summary from all sources |
| `prepare-meeting` | Compile results into meeting report |

### settings.json

Hook configuration. Tells Claude Code which hooks to run and when. Does not conflict with global `~/.claude/settings.json` -- different hook events are used.

## For other coding agents

These files are Claude Code specific. If you're using Gemini CLI, Codex, or another tool:

- Read `CLAUDE.md` at the repo root for project context
- Read `LAB.md` for experiment protocol
- Read `DISCOVERIES.md` for what's proven
- The locked files rule (harness.py, tracker.py, etc.) applies regardless of which agent you use -- it's in LAB.md rule #9

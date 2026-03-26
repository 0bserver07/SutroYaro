# Agent Infrastructure

How the workspace makes Claude Code (and other coding agents) effective at running research.

## The problem

A coding agent cloning this repo sees 150+ files. Without guidance, it reads random files, runs commands that break things, edits locked measurement code, and produces results it doesn't verify. It works, but slowly and with errors.

The agent infrastructure adds four things that change this: hooks that fire automatically at session boundaries, rules that constrain behavior, skills that define workflows, and a settings file that ties them together.

## What changed

Before this infrastructure, every session started with "catch me up" and the agent would spend 30-60 seconds reading files to figure out what was going on. Locked files (harness.py, tracker.py) had no enforcement -- the agent would sometimes edit them, breaking metric isolation. Experiments had no standard output format, so results were scattered across findings docs, results directories, and inline in commit messages.

After: the session-start hook shows project status automatically. The security guard blocks edits to locked files before they happen. The run-experiment skill defines a two-phase protocol that separates raw data from interpretation. The skill-forced-eval hook matches user requests to relevant skills so the agent doesn't forget they exist.

## Hooks

Three hooks run at different points in a Claude Code session. Each outputs JSON per the Claude Code protocol. They're Node.js scripts in `.claude/hooks/` with a shared utility library (`hook-common.cjs`).

**session-start.cjs** fires when a session begins. Shows git branch, uncommitted changes, open GitHub issues, last experiment result, and when Telegram/Google Docs were last synced. The agent starts with context instead of asking for it.

**security-guard.cjs** fires before every Bash, Edit, or Write tool call. Two tiers:

- Tier 1 (deny): blocks edits to locked measurement code (tracker.py, cache_tracker.py, data.py, config.py), `rm -rf` outside /tmp, and `git reset --hard`. These actions are stopped with no override.
- Tier 2 (confirm): intercepts `git push --force` and `git branch -D`. Returns a message telling the agent to ask the user first. The action is not blocked, but the agent is told to confirm.

harness.py is not locked by the hook because it needs legitimate edits when adding new methods. LAB.md rule #9 and PR review handle harness protection. A smarter approach could check whether the current branch is an experiment branch and only lock files during experiments.

**session-end.cjs** fires when a session ends. Shows uncommitted changes, recent commits, whether the tasks INDEX is stale, and unpushed commit count.

Skill activation is handled by the superpowers plugin (`using-superpowers` skill), which already tells the agent to check available skills on every message. No additional hook is needed for this.

### Hook events we don't use yet

Claude Code supports 26 hook events. We use 3 (SessionStart, PreToolUse, SessionEnd). Others that could be useful in future:

- `Notification` (matcher: `permission_prompt`): send a Telegram message when the agent is waiting for user approval. Enables async review without watching the terminal.
- `Stop`: prevent the agent from finishing before verification runs. Would have caught several issues in this session.
- `TaskCompleted`: block task completion if results haven't been verified.
- `PostToolUse`: log which tools the agent actually used, for understanding agent behavior across sessions.

Claude Code also supports HTTP hook types (POST to an endpoint) which could power a PR auto-review webhook: GitHub PR fires webhook, local server starts Claude Code review, review result goes to Telegram for approval. This is tracked in Issue #14.

## Rules

Two rule files in `.claude/rules/` that the agent loads as constraints.

**experiment-reproducibility.md**: every experiment must set and record the random seed, dump the full config to results JSON, record Python version and numpy version and OS, record the git commit hash, and include baseline comparison numbers. One variable at a time.

**agent-coordination.md**: defines when parallel agents are safe (non-overlapping files only), who owns which directories (experiment agents own `src/sparse_parity/experiments/` and `results/`, findings agents own `docs/findings/`, sync agents own `docs/google-docs/`), and that the main agent must verify all sub-agent outputs before committing.

## Skills

Six skills in `.claude/skills/`, each a directory with a SKILL.md and optional examples.

| Skill | What it does |
|-------|-------------|
| sutro-sync | Syncs Google Docs, Telegram, GitHub |
| sutro-context | Loads current project state from DISCOVERIES.md and recent discussions |
| anti-slop-guide | Reviews prose for AI writing patterns |
| run-experiment | Two-phase protocol: Phase 1 produces results JSON, Phase 2 produces findings doc |
| weekly-catchup | Generates weekly summary from all synced sources |
| prepare-meeting | Compiles experiment results into a meeting report |

The run-experiment skill enforces LAB.md rules #10 (two-phase results) and #11 (reproducibility). Phase 1 output goes to `results/{exp_id}/results.json` with raw numbers, config, and environment. Phase 2 output goes to `docs/findings/{exp_id}.md` with hypothesis, analysis, and impact. The separation means you can verify the numbers independently of the interpretation.

## How it connects to the eval environment

The eval environment (PR #49) tests whether an agent can navigate the research method space. The agent infrastructure makes the agent better at navigating it. They work together:

- The security guard prevents the agent from editing the harness, which the eval environment relies on for consistent measurements.
- The experiment reproducibility rule ensures eval results include seeds and environment info, so they can be compared across machines.
- The agent coordination rule prevents parallel agents from putting findings in the wrong directory (findings/ vs docs/findings/), which happened during the eval environment build.
- The run-experiment skill produces structured output that the eval answer key can reference.

## For other coding agents

The hooks and settings.json are Claude Code specific. Gemini CLI, Codex, and Antigravity do not run them.

The rules and skills are markdown files. Any agent that reads `.claude/rules/` and `.claude/skills/` can use them. The content (experiment protocol, file ownership, reproducibility requirements) applies regardless of which agent runs the experiments.

CLAUDE.md at the repo root contains the project context that all agents read. The `.claude/` directory adds enforcement and automation specific to Claude Code.

## Files

| File | What it is |
|------|-----------|
| `.claude/hooks/session-start.cjs` | Shows project status on session start |
| `.claude/hooks/security-guard.cjs` | Blocks locked file edits and destructive commands |
| `.claude/hooks/session-end.cjs` | Shows session summary on end |
| `.claude/hooks/hook-common.cjs` | Shared utility library for hooks |
| `.claude/rules/experiment-reproducibility.md` | Seed, config, environment recording requirements |
| `.claude/rules/agent-coordination.md` | Parallel dispatch, file ownership, review gates |
| `.claude/skills/run-experiment/SKILL.md` | Two-phase experiment protocol |
| `.claude/skills/weekly-catchup/SKILL.md` | Weekly summary workflow |
| `.claude/skills/prepare-meeting/SKILL.md` | Meeting report workflow |
| `.claude/settings.json` | Hook configuration (SessionStart, PreToolUse, SessionEnd) |
| `.claude/README.md` | File inventory for the .claude/ directory |

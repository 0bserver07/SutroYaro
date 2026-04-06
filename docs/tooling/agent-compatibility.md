# Agent compatibility

This project is designed for multiple coding agents to contribute autonomously.
Each agent vendor has their own convention for project-level config files.
This page tracks what works, what doesn't, and remaining gaps.

## Auto-loaded config files

| Agent | File | Auto-loaded | Verified |
|-------|------|-------------|----------|
| Claude Code | `CLAUDE.md` | Yes, at project root | ✅ |
| Codex CLI | `CODEX.md` | Yes, via `.codex/config.toml` | ✅ |
| Gemini CLI | `GEMINI.md` | Yes. Uses `@google/gemini-cli` (v0.35.3+). Not to be confused with the non-existent `@anthropic-ai/gemini-cli`. Runs via `bunx` or `npx`. | ✅ Runs, file verified present |
| Cursor | `.cursorrules` | Yes, editor extension | ❓ |
| Qwen Code | — | No documented convention | ❌ |
| Antigravity | — | Rewrites problems instead of following harness | ❌ |

## Why files diverge (no symlinks)

`CLAUDE.md` and `CODEX.md` are not identical. They have diverged because:

- Claude Code uses hooks/skills that Codex/Gemini don't have, so `CLAUDE.md` references them
- Codex has its own Telegram integration section (`telegram/tg-read.ts`) separate from `bin/tg-sync`
- Eval environment docs are only in `CLAUDE.md`

**Decision: do not symlink.** Each agent file is a curated subset relevant to that agent. When features are added, update all agent files that should reference them.

## Adding a new agent

1. Check if the agent auto-loads a project root file (check vendor docs)
2. If yes, create `FILENAME.md` matching the convention
3. If no, document how to inject context manually (initial prompt)
4. Update this page with the new row

## Known issues

- **Session-start hooks**: Only Claude Code has hook support. Other agents don't get the auto-status on launch. They can read `docs/tasks/INDEX.md` manually.
- **Security guard**: Claude Code hook blocks edits to locked files. Other agents must follow `LAB.md` rule #9 by convention.
- **Telegram context**: `bin/tg-sync` works for all agents (SQLite is universal). But only Claude Code agents get the Telegram context via session-start hooks.
- **Branch protection**: See [Issue #71](https://github.com/cybertronai/SutroYaro/issues/71). Without branch protection, agents can commit to main directly.

## Related issues

- [#13](https://github.com/cybertronai/SutroYaro/issues/13): Agent compatibility layer (this page)
- [#71](https://github.com/cybertronai/SutroYaro/issues/71): Enable branch protection on main
- [#28](https://github.com/cybertronai/SutroYaro/issues/28): Formalize slash commands

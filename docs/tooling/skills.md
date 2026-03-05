# Skills & Plugins

Claude Code supports skills (reusable workflow templates) and plugins (installable packages of skills). Here's what we use in this project.

## Plugin: Superpowers

[obra/superpowers](https://github.com/obra/superpowers) — an agentic skills framework that adds a 7-phase development workflow to Claude Code. In the Anthropic marketplace since Jan 2026. Installed as a Claude Code plugin.

The skills auto-invoke when relevant — you don't need to call them manually.

| Skill | When it fires | What it does |
|-------|--------------|-------------|
| **Brainstorming** | Before any creative work | Explores intent, requirements, and design before writing code |
| **Writing Plans** | Multi-step tasks | Write a plan, get approval, then execute with checkpoints |
| **Executing Plans** | After plan approval | Follows the plan step-by-step with review gates |
| **Test-Driven Development** | Before implementation | Write tests first, then code to pass them |
| **Systematic Debugging** | Any bug or test failure | 4-phase root cause analysis: trace, hypothesize, verify, fix |
| **Dispatching Parallel Agents** | 2+ independent tasks | Spawns multiple Claude Code agents working simultaneously |
| **Subagent-Driven Development** | Implementation plans with independent tasks | Coordinates agents within a single session |
| **Verification Before Completion** | Before claiming "done" | Requires running commands and confirming output before success claims |
| **Requesting Code Review** | After completing a step | Reviews implementation against plan, reports issues by severity |
| **Receiving Code Review** | When given feedback | Prevents blind agreement — requires verifying feedback is technically correct |
| **Using Git Worktrees** | Feature work needing isolation | Creates isolated git worktrees for parallel development |
| **Finishing a Development Branch** | All tests pass, ready to integrate | Presents merge/PR/keep/discard options, cleans up worktree |
| **Writing Skills** | Creating or editing skills | Guides skill authoring and verification |

Related repos:

- [obra/superpowers-skills](https://github.com/obra/superpowers-skills) — community-editable skills, cloned to `~/.config/superpowers/skills/`
- [obra/superpowers-lab](https://github.com/obra/superpowers-lab) — experimental skills (semantic duplication detection, etc.)
- [obra/superpowers-marketplace](https://github.com/obra/superpowers-marketplace) — curated plugin marketplace

## Custom Skills (included in this repo)

This repo ships project-level skills in `.claude/skills/`. Anyone cloning the repo gets them automatically — Claude Code detects them on session start.

```
.claude/skills/
  anti-slop-guide/
    SKILL.md          # auto-detected by Claude Code
```

Skills can also live at the user level (`~/.claude/skills/`) for use across all projects.

### Anti-Slop Guide

**Location**: `.claude/skills/anti-slop-guide/SKILL.md` (in this repo)
**Invoked**: `/anti-slop-guide` or automatically when writing prose

Detects and removes AI writing patterns from prose. We ran it on all 32 MkDocs pages after initial generation — the difference was large. Pages went from sounding like ChatGPT marketing copy to reading like research notes.

See the [full reference](anti-slop-guide.md).

Sources: [stop-slop](https://github.com/hardikpandya/stop-slop) by Hardik Pandya, [Wikipedia: Signs of AI Writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), and several community guides. MIT licensed.

### Ralph Wiggum

A technique for maintaining persistent background loops in Claude Code sessions. Available commands: `/ralph-wiggum:help`, `/ralph-wiggum:ralph-loop`, `/ralph-wiggum:cancel-ralph`.

## MCP Servers

Model Context Protocol servers extend Claude Code with new tools.

### Currently Active

| Server | What it provides |
|--------|-----------------|
| **drawio-mcp** | Create and edit diagrams (`.drawio.svg`) from Claude Code |
| **macos-control** | Screenshot, mouse/keyboard control, screen automation |
| **macos-automator** | AppleScript/JXA execution, app automation |
| **apple-calendar** | Calendar event management |
| **macos-ui-automation** | Accessibility-based UI element interaction |

### Worth Adding

| Server | Why |
|--------|-----|
| **Google Docs MCP** | Live read/write access to Google Docs (replace sync script) |
| **Browser MCP** | Fetch web content, papers, arxiv results |
| **GitHub MCP** | Direct PR/issue management without `gh` CLI |

## Skills Worth Building

These don't exist yet but would fit the Sutro research workflow:

**Research Sprint** — enforce the loop from [prompting strategies](../findings/prompting-strategies.md): literature search, gap diagnosis, ranked experiments, one-at-a-time execution, failure analysis.

**Energy Audit** — run ARD analysis on an experiment script, compare against baseline, identify top memory access bottlenecks.

**Google Docs Sync** — wrap `sync_google_docs.py`: pull latest, diff, update cross-references, rebuild site.

## More Skills & Resources

- [travisvn/awesome-claude-skills](https://github.com/travisvn/awesome-claude-skills) — curated list of Claude skills and tools
- [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills) — 500+ agent skills from official dev teams and community
- [hesreallyhim/awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code) — skills, hooks, slash-commands, agent orchestrators

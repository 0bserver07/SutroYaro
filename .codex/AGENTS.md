# Codex Project Instructions

These instructions apply to all Codex CLI sessions in this repo.

## Before Any Research Task

Load current project state before running experiments, reviewing PRs, or writing findings.

1. Read these files in order:
   - **CODEX.md** -- project context, current best methods, constraints
   - **DISCOVERIES.md** -- what's proven, what failed, open questions (bottom of file)
   - **AGENT.md** -- machine-executable experiment loop (if running autonomous)
   - **LAB.md** -- experiment protocol, rules (especially rule #9: metric isolation)

2. Check recent Telegram activity (if synced):

```python
import json
for f in ['chat-yad.json', 'chat-yaroslav.json', 'challenge-1-sparse-parity.json']:
    path = f'src/sparse_parity/telegram_sync/{f}'
    try:
        msgs = json.load(open(path))
        print(f'\n=== {f} (last 3) ===')
        for m in msgs[:3]:
            print(f"  [{m['date'][:10]}] {m['sender']}: {m['text'][:150]}")
    except FileNotFoundError:
        print(f'{f} not found -- run: bun run sync_telegram.ts')
```

3. Check GitHub for open work:

```bash
gh pr list --repo cybertronai/SutroYaro --state open
gh issue list --repo cybertronai/SutroYaro --state open
```

4. Before writing code, check:
   - `research/search_space.yaml` for allowed parameter ranges
   - `research/questions.yaml` for the dependency graph of open questions

## Current State

| Fact | Value |
|------|-------|
| Best method | GF(2) Gaussian elimination, 509us, ARD ~500 |
| Best energy proxy | DMC (Data Movement Complexity, Ding et al.) |
| Experiments done | 33+ (see `research/log.jsonl`) |
| Open questions | Bottom of DISCOVERIES.md (Q7, Q11-Q13 still open) |
| Next milestone | Energy-efficient nanoGPT training ("final exam") |
| Meeting cadence | Mondays 18:00 at South Park Commons |

## Sync Routine

Run at session start and before any push:

```bash
# Telegram (daily)
bun run sync_telegram.ts

# Or use the targeted read/send scripts:
bun telegram/tg-read.ts --topic "General" --limit 10
bun telegram/tg-send.ts --topic "agents" --message "Status update"

# Google Docs (weekly, after Monday meetings)
python3 src/sync_google_docs.py

# GitHub
gh pr list --repo cybertronai/SutroYaro --state open
gh issue list --repo cybertronai/SutroYaro --state open
```

Before pushing:
1. Update `docs/changelog.md` (bump version)
2. `python3 -m mkdocs build` to verify no broken links
3. Show the diff and wait for approval before `git push`

## Writing Rules (Anti-Slop)

Apply these to all prose (findings docs, DISCOVERIES.md updates, PR descriptions):

1. Cut filler phrases. Say the thing directly.
2. Break formulaic structures. No binary contrasts, no dramatic fragmentation.
3. Vary rhythm. Mix sentence lengths. Two items beat three.
4. Trust readers. State facts directly.
5. Prefer plain verbs. "used" not "leveraged," "showed" not "showcased."
6. Use simple copulatives. Write "X is Y" not "X serves as Y."
7. Kill em dashes. Use commas or periods.
8. Never triple. Two items in a list, not three.
9. Be specific. Replace generic statements with concrete details.
10. No AI vocabulary: delve, tapestry, landscape, pivotal, showcase, testament, underscore, foster, garner, interplay, intricate, vibrant, robust, seamless, paramount, multifaceted, nuanced, groundbreaking, cornerstone, transformative, synergy.

Full guide: `.claude/skills/anti-slop-guide/SKILL.md` (plain markdown, readable by any tool).

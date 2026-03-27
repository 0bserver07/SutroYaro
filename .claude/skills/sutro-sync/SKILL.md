---
name: sutro-sync
description: Use at the start of any session and before pushing. Syncs Telegram, Google Docs, and GitHub state for the Sutro Group research workspace.
---

# Sutro Sync Routine

Run this at session start and before any push. Covers three sources: Telegram, Google Docs, GitHub.

## 1. Telegram (daily)

```bash
bin/tg-sync
```

Syncs 6 forum topics to a local SQLite database (`telegram.db`). First run does a full backfill; subsequent runs are incremental (only new messages).

Query recent messages from priority topics to check for new questions, directions, or results:

```bash
sqlite3 telegram.db "SELECT date, sender, substr(text, 1, 150) FROM messages
  WHERE topic_id IN (SELECT id FROM topics WHERE slug IN ('chat-yad', 'chat-yaroslav', 'challenge-1-sparse-parity'))
  AND date > datetime('now', '-3 days')
  ORDER BY date DESC LIMIT 15"
```

Report anything relevant to the user.

**Fallback**: If `telegram.db` doesn't exist and credentials aren't configured, read the committed JSON files in `src/sparse_parity/telegram_sync/` instead (may be stale).

**Prerequisites**: `bun install`, `.env` with `TELEGRAM_API_ID` and `TELEGRAM_API_HASH`, `tg auth login` (one-time). Full setup: [docs/tooling/telegram-setup.md](docs/tooling/telegram-setup.md).

## 2. Google Docs (weekly, after Monday meetings)

```bash
python3 src/sync_google_docs.py
```

Pulls 15+ Google Docs to `docs/google-docs/`. After syncing:

1. Check `docs/google-docs/sutro-group-main.md` for new meeting doc links
2. If new docs found, add them:
   ```bash
   python3 src/sync_google_docs.py --add "URL" "name" "Description"
   python3 src/sync_google_docs.py   # re-run
   ```
3. Add cross-reference headers (`!!! info` admonition) to new docs
4. Add new pages to `mkdocs.yml` nav under Meetings > Google Docs
5. Update `docs/meetings/index.md` and `docs/meetings/notes.md`

**Prerequisites**: `pandoc` (`brew install pandoc`). Docs must be "Anyone with the link" sharing.

## 3. GitHub (daily)

```bash
gh pr list --repo cybertronai/SutroYaro --state open
gh issue list --repo cybertronai/SutroYaro --state open
```

Check for new PRs from contributors (Andy/zh4ngx, Michael, others). Review code, verify results are reproducible, check that DISCOVERIES.md is updated.

## 4. Staleness check (before pushing)

Check `docs/index.md` (homepage) for stale numbers and missing features:

```python
import re

# Check experiment count matches DISCOVERIES.md
with open('DISCOVERIES.md') as f:
    disc = f.read()
exp_count = len(re.findall(r'^\| exp_', disc, re.MULTILINE))

with open('docs/index.md') as f:
    index = f.read()

# Flag if homepage says fewer experiments than DISCOVERIES.md has
if f'{exp_count}' not in index:
    print(f'WARNING: homepage may be stale. DISCOVERIES.md has {exp_count} experiments.')

# Check challenge count
challenges = ['sparse-parity', 'sparse-sum', 'sparse-and']
for c in challenges:
    if c not in index:
        print(f'WARNING: homepage missing challenge: {c}')
```

Also check:
- Does the homepage mention all `bin/` scripts?
- Are the "Where to Find Things" links still valid?
- Does the experiment count match?

If anything is stale, update `docs/index.md` before pushing.

## Before pushing

1. Run staleness check above
2. Update `docs/changelog.md` (bump version)
3. `python3 -m mkdocs build` to verify no broken links
4. Show the user the diff and wait for approval before `git push`
5. Deploy: `python3 -m mkdocs gh-deploy --force`

## Tool-agnostic notes

This skill works with any AI coding tool that reads files:

- **Claude Code**: reads `.claude/skills/` automatically
- **Cursor/Windsurf**: paste the commands or add to `.cursorrules`
- **Codex/other**: include this file in the system prompt or project context
- **Manual**: follow the checklists in `docs/tooling/sync-runbook.md`

All state lives in files (JSON, markdown, YAML). No tool-specific APIs needed.

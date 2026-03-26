---
name: weekly-catchup
description: Use at the start of a weekly session. Syncs all sources and generates a catch-up summary.
---

# Weekly Catch-Up

## Steps

1. Sync Google Docs. Run `python3 src/sync_google_docs.py`. Check for new meeting docs.

2. Sync Telegram. Run `bun run sync_telegram.ts`. Check chat-yad, chat-yaroslav, challenge channels.

3. Check GitHub. Run `gh issue list --repo cybertronai/SutroYaro --state open` and `gh pr list`.

4. Read the last catch-up in `docs/catchups/` for context on what happened last week.

5. Generate the catch-up page. Create `docs/catchups/YYYY-MM-DD.md` with:
   - Sync status (docs synced, messages count, issues/PRs)
   - New ideas from Telegram (summarize, don't quote private messages)
   - GitHub issues update (new, closed, in progress)
   - What's due (upcoming meetings, deadlines)
   - Action items for the week

6. Update `docs/catchups/index.md` with the new entry.

7. Update `mkdocs.yml` nav if needed.

## Checklist

- [ ] Google Docs synced
- [ ] Telegram synced
- [ ] GitHub checked
- [ ] Catch-up page created
- [ ] Index updated
- [ ] Nav updated

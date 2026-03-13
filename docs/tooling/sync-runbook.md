# Sync Runbook

Checklists for keeping the knowledge base current. Three cadences: weekly after meetings, daily for discussion threads, and per-session for context.

## Weekly (after Monday meeting)

Run this Tuesday morning or whenever the meeting notes are posted.

### 1. Pull Google Docs

```bash
python3 src/sync_google_docs.py
```

### 2. Check for new docs

Open the synced main page (`docs/google-docs/sutro-group-main.md`) and look for new meeting doc links. If there are new ones:

```bash
python3 src/sync_google_docs.py --add "https://docs.google.com/document/d/DOC_ID/edit" "name" "Description"
python3 src/sync_google_docs.py   # re-run to pull the new doc
```

### 3. Add cross-reference headers

Each new doc in `docs/google-docs/` needs an admonition at the top:

```markdown
!!! info "Cross-references"
    **Source**: [Google Doc](URL) · [Meeting #N summary](../meetings/notes.md#anchor)
    **Related**: [other doc](other-doc.md)
```

The sync script preserves these on re-sync.

### 4. Wire into MkDocs

- Add new pages to `mkdocs.yml` under `Meetings > Google Docs`
- Add a row to `docs/meetings/index.md` schedule table
- Add a section to `docs/meetings/notes.md` with meeting summary

### 5. Update changelog

Add a version entry to `docs/changelog.md`.

### 6. Build, commit, deploy

```bash
python3 -m mkdocs build          # verify no broken links
git add -A docs/ mkdocs.yml src/docs_config.json
git commit -m "Sync Meeting #N docs, update nav"
git push origin main
python3 -m mkdocs gh-deploy --force
```

## Daily

### Telegram

```bash
bun run sync_telegram.ts
```

Output goes to `src/sparse_parity/telegram_sync/messages.json` (gitignored). Read it to check for new questions, results, or discussion that should inform the next session.

### GitHub

```bash
gh pr list --repo 0bserver07/SutroYaro
gh issue list --repo 0bserver07/SutroYaro
```

Review and respond to any new PRs or issues. See [automation docs](automation.md) for setup details.

## Per-session (start of work)

These happen automatically when Claude Code reads CLAUDE.md, but worth checking manually:

1. `git status` for uncommitted sync data
2. Quick Telegram sync if the group may have posted
3. `gh pr list` if collaborators (Andy, Michael) may have pushed

## Current sync config

To see what docs are configured:

```bash
python3 src/sync_google_docs.py --list
```

As of Meeting #8, we sync 15 Google Docs (meetings 1-2, 6-8, sprints, challenge spec, and supporting docs). Meetings 3-5 are PDFs on Google Drive and can't be pulled via the export endpoint.

# Automation Scripts

## sync_google_docs.py

Pulls Google Docs to local markdown files and extracts all hyperlinks into a references page.

### Usage

```bash
python3 src/sync_google_docs.py              # sync all configured docs
python3 src/sync_google_docs.py --list       # show configured docs
python3 src/sync_google_docs.py --add URL NAME  # add a new doc
```

### Requirements

- `pandoc` (install with `brew install pandoc`)
- Google Doc must be publicly accessible (Share > Anyone with the link)

### How It Works

1. Downloads the Google Doc as HTML via the export URL (`/export?format=html`)
2. Converts HTML to markdown using pandoc
3. Cleans up the markdown (removes excessive whitespace, fixes link formatting)
4. Extracts all hyperlinks and saves them to `docs/references_auto.md`
5. Saves the cleaned markdown to `docs/google-docs/{name}.md`

### Configuration

Docs are configured in `src/docs_config.json`:

```json
[
  {
    "url": "https://docs.google.com/document/d/DOCID/edit",
    "name": "meeting-7-notes",
    "description": "Meeting #7: Sparse parity discussion (02 Mar 2026)"
  }
]
```

### Limitations

- Only works with Google Docs (not Sheets, Slides, or PDFs on Drive)
- Meetings 3-5 link to PDFs on Google Drive, so they can't be pulled this way
- Embedded images are lost (replaced with `[embedded image]`)
- Tables from Google Docs sometimes convert poorly

### After Syncing

After pulling new docs:

1. Add the new pages to `mkdocs.yml` nav
2. Add cross-reference headers (the `!!! info` admonition boxes)
3. Update `docs/meetings/index.md` and `docs/meetings/notes.md` with local links
4. Run `python3 -m mkdocs build` to verify

See the [sync runbook](sync-runbook.md) for weekly/daily/per-session checklists.

## sync_telegram.ts

Pulls messages from a Telegram group topic thread into a local JSON file. Used to sync the Sutro Group's "Sparse Parity" discussion thread.

### Prerequisites

1. **Bun** (TypeScript runtime): `curl -fsSL https://bun.sh/install | bash`
2. **tg CLI** (Telegram auth session): `bun install -g @goodit/telegram-sync-cli`
3. **Telegram API credentials**: Get `api_id` and `api_hash` from [my.telegram.org/apps](https://my.telegram.org/apps)

### First-Time Setup

```bash
# 1. Install dependencies
cd /path/to/SutroYaro
bun install

# 2. Create .env file with your Telegram API credentials
cp .env.example .env
# Edit .env and fill in TELEGRAM_API_ID and TELEGRAM_API_HASH

# 3. Authenticate with Telegram (one-time)
tg auth login
# Follow the prompts: enter phone number, then the code from Telegram
```

The `tg auth login` command creates a session file at `~/.telegram-sync-cli/session_1.db`. The sync script reuses this session, so you only need to authenticate once.

### Usage

```bash
bun run sync_telegram.ts
```

Output:

```
Connecting to Telegram...
Connected.
Resolved sutro_group -> {...}
Fetching forum topics...
Found topic: [656] "Challenge 1: Sparse Parity"
Fetching messages...
  fetched 31 messages so far...

Done. 31 messages written to src/sparse_parity/telegram_sync/messages.json
```

### Configuration

The script has three constants at the top of `sync_telegram.ts`:

| Constant | Default | What it does |
|----------|---------|-------------|
| `CHANNEL_USERNAME` | `"sutro_group"` | The Telegram group to sync from |
| `TOPIC_KEYWORD` | `"sparse parity"` | Substring match against forum topic titles |
| `OUTPUT_FILE` | `src/sparse_parity/telegram_sync/messages.json` | Where the JSON gets written |

To sync a different topic, change `TOPIC_KEYWORD`. To sync a different group, change `CHANNEL_USERNAME`.

### Output Format

The script writes an array of message objects:

```json
[
  {
    "id": 735,
    "date": "2026-03-06T20:56:06.000Z",
    "sender": "G B",
    "text": "yeah I think we discussed trying to get a single formula...",
    "replyTo": 656
  }
]
```

Messages are ordered newest-first. The `replyTo` field points to the topic's root message ID (656 for the sparse parity thread).

### How It Works

1. Connects to Telegram using the MTProto client (`@mtcute/bun`) with the session from `tg auth login`
2. Resolves the channel by username
3. Fetches forum topics and finds the one matching `TOPIC_KEYWORD`
4. Paginates through all replies in that topic thread (100 messages per batch, 500ms rate limit between batches)
5. Resolves user IDs to display names
6. Writes the JSON to the output directory

### Troubleshooting

| Problem | Fix |
|---------|-----|
| "Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env" | Create `.env` from `.env.example` and fill in your credentials |
| "Session not found" | Run `tg auth login` to authenticate |
| "Topic matching 'sparse parity' not found" | The script prints available topics. Check the spelling or update `TOPIC_KEYWORD` |
| "FLOOD_WAIT" error | Telegram rate limit. Wait the indicated seconds and retry |

### Privacy

The `messages.json` file contains real messages from group members. The `.env` file contains API credentials. Both are gitignored. The sync script itself is safe to commit.

## References Auto-Extraction

The sync script also builds `docs/references_auto.md` — a flat list of all unique URLs found across all pulled Google Docs. This feeds into the curated `docs/references.md` page which organizes them by category (Google Docs, Drive, Colab, GitHub, Gemini, Other).

## export_sessions.py

Exports Claude Code session traces from `~/.claude` into `.traces/sessions/` as readable text files.

### Why

Claude Code stores every conversation as JSONL files in `~/.claude/projects/`. These contain the full back-and-forth of every research session — hypotheses tested, code written, experiments run, dead ends hit. Without exporting them, this research history is invisible and eventually lost when sessions age out or the machine changes.

The exported traces let us:

- Review what each agent actually did during parallel experiments (sparse-parity team had 4 agents, research-loop had 5, blank-slate had 3)
- Find abandoned ideas worth revisiting
- See which approaches were tried and why they failed
- Reconstruct the reasoning behind decisions in DISCOVERIES.md

### Usage

```bash
python3 .traces/export_sessions.py              # export all sessions
python3 .traces/export_sessions.py --list       # list sessions with metadata
python3 .traces/export_sessions.py SESSION_ID   # export one session
python3 .traces/export_sessions.py --team sparse-parity  # export one team
```

### How It Works

1. Reads JSONL files from `~/.claude/projects/-Users-yadkonrad-dev-dev-year26-feb26-SutroYaro/`
2. Parses the nested message format (`entry.message.role`, `entry.message.content`)
3. Extracts text from content blocks, summarizes tool calls (Read, Write, Edit, Bash, Grep, Agent, etc.)
4. Strips `<system-reminder>` tags and skips system messages
5. Writes readable text files to `.traces/sessions/` with `YOU` / `CLAUDE` labels
6. Generates `INDEX.md` with a table of all exported sessions

Filenames include team and agent names when available: `sparse-parity-metrics-agent-04f577d0.txt`.

### Privacy

The `.traces/sessions/` directory is gitignored. The export script is committed, the outputs are not. Session traces contain raw conversation data and should stay local.

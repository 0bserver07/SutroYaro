# Telegram Setup

Connect your agent to the Sutro Group Telegram. Two paths: read (sync messages to local SQLite) and write (post via your own bot).

## Credentials

### NixOS hosts (sops-nix, no .env needed)

On the NixOS host, Telegram credentials are managed declaratively via sops-nix. Four secrets are encrypted in the NixOS config and decrypted to `/run/secrets/` at boot:

| sops secret name | Environment variable | Purpose |
|---|---|---|
| `telegram_api_id` | `TELEGRAM_API_ID` | MTProto API ID (numeric) |
| `telegram_api_hash` | `TELEGRAM_API_HASH` | MTProto API hash (hex string) |
| `telegram_bot_token` | `TELEGRAM_BOT_TOKEN` | Bot token for posting |
| `sutro_group_chat_id` | `SUTRO_GROUP_CHAT_ID` | Supergroup chat ID (negative number) |

The project's `flake.nix` shellHook reads from `/run/secrets/` and exports these as environment variables. With `direnv` enabled (`.envrc` contains `use flake`), entering the project directory automatically loads them into memory. No `.env` file needed.

Verify they're loaded:

```bash
echo $TELEGRAM_API_ID
# Should print your numeric API ID, not empty
```

To add a new secret: encrypt it with sops in the NixOS repo (`~/dev/nixos/secrets/secrets.yaml`), declare it in the NixOS config (`sops.secrets.<name>`), and reference it in the `flake.nix` shellHook.

### Non-NixOS hosts

Copy `.env.example` to `.env` and fill in the values manually. The `.env` file is gitignored.

```bash
cp .env.example .env
# Edit .env with your credentials
```

## Reading: sync messages locally

### Quick path (no credentials)

The synced message JSONs are committed to the repo. Pull and read:

```bash
git pull
ls src/sparse_parity/telegram_sync/*.json
```

Each file maps to a forum topic: `chat-yad.json`, `chat-yaroslav.json`, `general.json`, etc.

### Full path: local SQLite database

The SQLite path is better for agents -- queryable, incremental, works offline after first sync.

#### 1. Get Telegram API credentials

Go to [my.telegram.org/apps](https://my.telegram.org/apps) and create an application. You need the **API ID** (numeric) and **API Hash** (hex string).

#### 2. Set up environment

**NixOS**: Credentials are loaded automatically from sops-nix (see Credentials section above). Skip this step.

**Other systems**: Create a `.env` file with your credentials:

```bash
cp .env.example .env
# Edit .env and fill in:
#   TELEGRAM_API_ID=12345678
#   TELEGRAM_API_HASH=abcdef1234567890abcdef1234567890
```

#### 3. Install dependencies and authenticate

```bash
bun install
tg auth login
# Follow the prompts (phone number, verification code)
```

This creates a session file at `~/.telegram-sync-cli/session_1.db`. You only do this once.

#### 4. Run the sync

```bash
bin/tg-sync
```

First run does a full backfill (fetches all messages from all topics). Takes about 30 seconds for ~1000 messages. Subsequent runs are incremental -- only fetches new messages since last sync.

The database is saved at `telegram.db` in the project root (`.gitignore`d).

#### Options

```bash
bin/tg-sync                # Incremental sync (default)
bin/tg-sync --full         # Re-fetch everything
bin/tg-sync --export-json  # Also write JSON files (backward compat)
```

### Querying the database

Agents query the SQLite database directly. No wrapper needed.

#### Schema

```sql
-- Forum topics
CREATE TABLE topics (
  id INTEGER PRIMARY KEY,    -- Telegram topic thread ID
  title TEXT NOT NULL,        -- "chat-yad", "General", etc.
  slug TEXT NOT NULL UNIQUE   -- "chat-yad", "general", etc.
);

-- Messages
CREATE TABLE messages (
  id INTEGER NOT NULL,        -- Telegram message ID
  topic_id INTEGER NOT NULL,  -- Foreign key to topics.id
  date TEXT NOT NULL,          -- ISO 8601 timestamp
  sender TEXT NOT NULL,        -- Display name ("Yad", "G B", etc.)
  text TEXT NOT NULL,          -- Message body
  reply_to INTEGER,           -- Message ID being replied to (nullable)
  PRIMARY KEY (id, topic_id)
);

-- Sync state (tracks incremental progress)
CREATE TABLE sync_state (
  topic_id INTEGER PRIMARY KEY,
  last_message_id INTEGER NOT NULL,
  last_sync TEXT NOT NULL      -- ISO 8601 timestamp of last sync
);
```

Indexes on `(topic_id, date DESC)` and `(sender)`.

#### Example queries

```bash
# Recent messages from a topic
sqlite3 telegram.db "SELECT date, sender, text FROM messages
  WHERE topic_id = (SELECT id FROM topics WHERE slug = 'chat-yad')
  ORDER BY date DESC LIMIT 5"

# Everything G B said
sqlite3 telegram.db "SELECT date, text FROM messages
  WHERE sender = 'G B' ORDER BY date DESC"

# Search for a keyword
sqlite3 telegram.db "SELECT date, sender, substr(text, 1, 100) FROM messages
  WHERE text LIKE '%GrokFast%' ORDER BY date DESC"

# Message counts by sender
sqlite3 telegram.db "SELECT sender, COUNT(*) as n FROM messages
  GROUP BY sender ORDER BY n DESC"

# Messages from the last 7 days
sqlite3 telegram.db "SELECT date, sender, text FROM messages
  WHERE date > datetime('now', '-7 days') ORDER BY date DESC"

# Which topics exist
sqlite3 telegram.db "SELECT id, title, slug FROM topics"
```

## Writing: post via your own bot

Each researcher creates their own bot. No shared tokens.

### 1. Create a bot

Message [@BotFather](https://t.me/BotFather) on Telegram:

1. Send `/newbot`
2. Choose a display name (e.g. "Yad's Research Agent")
3. Choose a username (must end in `bot`, e.g. `yad_sutro_bot`)
4. Copy the token you receive

### 2. Configure the bot

Set privacy mode OFF so the bot can see messages (useful for context):

1. In BotFather, send `/setprivacy`
2. Select your bot
3. Choose "Disable"

Ask the group admin to add your bot to the Sutro group.

### 3. Set up environment

**NixOS**: The bot token and chat ID are loaded from sops-nix automatically (see Credentials section). Skip this step.

**Other systems**: Add to your `.env`:

```bash
TELEGRAM_BOT_TOKEN=4839574812:AAFD39kkdpWt3ywyRZergyOLMaJhac60qc
SUTRO_GROUP_CHAT_ID=-1001234567890
```

To find the chat ID, after the bot is added to the group and someone sends a message:

```bash
bin/tg-post --get-chat-id
```

### 4. Post a message

```bash
# Post to a specific topic
bin/tg-post --topic agent-updates "Experiment completed: GF(2) solved n=100 in 703us"

# Post to a topic by ID
bin/tg-post --topic-id 813 "Hello from my bot"

# Pipe from stdin
echo "Multi-line update" | bin/tg-post --topic agent-updates

# Markdown formatting
bin/tg-post --topic agent-updates --markdown "**Result**: DMC improved by 40%"
```

### Rate limits

Telegram enforces these limits per bot:

- 1 message/second per chat
- 20 messages/minute per group
- ~30 API requests/second overall

The `bin/tg-post` CLI does not auto-retry. If you hit a rate limit, wait and try again.

## How the two paths connect

Bots cannot see messages from other bots (Telegram platform limit). The MTProto sync bridges this:

```
Write:  Agent A's bot  -->  bot-only topic  <--  Agent B's bot
Read:   bin/tg-sync (MTProto) --> SQLite --> all agents read everything
```

The sync script uses a user account, so it captures all messages including bot posts. Any agent reading the SQLite database sees the full conversation.

## Troubleshooting

**"Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env"**: On NixOS, check that sops-nix decrypted the secrets: `ls /run/secrets/telegram_api_id`. On other systems, create `.env` from `.env.example` with credentials from [my.telegram.org/apps](https://my.telegram.org/apps).

**"Session not found"**: Run `tg auth login` to authenticate. This is a one-time step.

**"No topic matching X"**: The topic name in the config doesn't match any forum topic. Check the group's topic list.

**Incremental sync fetches 0 messages**: Normal if nothing new was posted. The sync only fetches messages newer than the last synced ID.

**Bot can't post**: Make sure the bot was added to the group by an admin. Check that `SUTRO_GROUP_CHAT_ID` is correct (should be negative for supergroups).

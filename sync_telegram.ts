import { TelegramClient } from "@mtcute/bun";
import { tl } from "@mtcute/tl";
import { writeFileSync, existsSync, mkdirSync } from "fs";
import { join, resolve } from "path";
import { homedir } from "os";

// --- Config ---
const CHANNEL_USERNAME = "sutro_group";

// Topics to sync, in priority order. Use "all" to sync every topic.
// Each entry: keyword substring to match against topic title.
const TOPICS_TO_SYNC = [
  "chat-yad",
  "chat-yaroslav",
  "challenge #1: sparse parity",
  "General",
  "In-person meetings",
  "Introductions",
];

const OUTPUT_DIR = resolve(import.meta.dir, "src/sparse_parity/telegram_sync");

// Reuse the tg CLI session directly
const SESSION_PATH = join(homedir(), ".telegram-sync-cli", "session_1.db");

const API_ID = parseInt(process.env.TELEGRAM_API_ID ?? "0", 10);
const API_HASH = process.env.TELEGRAM_API_HASH ?? "";

if (!API_ID || !API_HASH) {
  console.error("Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env");
  process.exit(1);
}

if (!existsSync(SESSION_PATH)) {
  console.error(`Session not found at ${SESSION_PATH}. Run 'tg auth login' first.`);
  process.exit(1);
}

const client = new TelegramClient({
  apiId: API_ID,
  apiHash: API_HASH,
  storage: SESSION_PATH,
});

function slugify(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

async function fetchTopicMessages(
  resolved: tl.TypeInputPeer,
  topicId: number,
  users: Map<number, string>
): Promise<tl.RawMessage[]> {
  const allMessages: tl.RawMessage[] = [];
  let offsetId = 0;
  const BATCH_SIZE = 100;

  while (true) {
    const history = await client.call({
      _: "messages.getReplies",
      peer: resolved,
      msgId: topicId,
      offsetId,
      offsetDate: 0,
      addOffset: 0,
      limit: BATCH_SIZE,
      maxId: 0,
      minId: 0,
      hash: 0,
    });

    const resp = history as tl.RawMessagesChannelMessages;

    // Collect users from this response
    if ("users" in resp) {
      for (const u of resp.users) {
        if (u._ === "user") {
          users.set(u.id, [u.firstName, u.lastName].filter(Boolean).join(" "));
        }
      }
    }

    const msgs = resp.messages.filter(
      (m): m is tl.RawMessage => m._ === "message"
    );

    if (msgs.length === 0) break;

    allMessages.push(...msgs);
    offsetId = msgs[msgs.length - 1].id;

    // Rate limit courtesy
    await new Promise((r) => setTimeout(r, 500));
  }

  return allMessages;
}

async function main() {
  console.log("Connecting to Telegram...");
  await client.start();
  console.log("Connected.");

  const resolved = await client.resolvePeer(CHANNEL_USERNAME);
  console.log(`Resolved ${CHANNEL_USERNAME}`);

  const inputChannel: tl.TypeInputChannel = {
    _: "inputChannel",
    channelId: (resolved as tl.RawInputPeerChannel).channelId,
    accessHash: (resolved as tl.RawInputPeerChannel).accessHash,
  };

  console.log("Fetching forum topics...");
  const topics = await client.call({
    _: "channels.getForumTopics",
    channel: inputChannel,
    limit: 100,
    offsetDate: 0,
    offsetId: 0,
    offsetTopic: 0,
  });

  const forumTopics = (topics as tl.RawMessagesForumTopics).topics.filter(
    (t): t is tl.RawForumTopic => t._ === "forumTopic"
  );

  console.log(
    `Found ${forumTopics.length} topics: ${forumTopics.map((t) => t.title).join(", ")}`
  );

  // Collect users from topics response
  const users = new Map<number, string>();
  if ("users" in topics) {
    for (const u of (topics as tl.RawMessagesForumTopics).users) {
      if (u._ === "user") {
        users.set(u.id, [u.firstName, u.lastName].filter(Boolean).join(" "));
      }
    }
  }

  // Match topics to sync
  const toSync: tl.RawForumTopic[] = [];
  for (const keyword of TOPICS_TO_SYNC) {
    const match = forumTopics.find(
      (t) => t.title.toLowerCase() === keyword.toLowerCase()
    );
    if (match) {
      toSync.push(match);
    } else {
      console.warn(`  Warning: no topic matching "${keyword}"`);
    }
  }

  if (toSync.length === 0) {
    console.error("No matching topics found.");
    process.exit(1);
  }

  if (!existsSync(OUTPUT_DIR)) mkdirSync(OUTPUT_DIR, { recursive: true });

  let totalMessages = 0;

  for (const topic of toSync) {
    const slug = slugify(topic.title);
    const outFile = join(OUTPUT_DIR, `${slug}.json`);

    console.log(`\nSyncing [${topic.id}] "${topic.title}" -> ${slug}.json`);

    const messages = await fetchTopicMessages(resolved, topic.id, users);

    const output = messages.map((m) => ({
      id: m.id,
      date: new Date(m.date * 1000).toISOString(),
      sender:
        users.get(
          Number(m.fromId && "userId" in m.fromId ? m.fromId.userId : 0)
        ) ?? String(m.fromId),
      text: m.message,
      replyTo:
        m.replyTo && "replyToMsgId" in m.replyTo
          ? m.replyTo.replyToMsgId
          : null,
    }));

    writeFileSync(outFile, JSON.stringify(output, null, 2));
    console.log(`  ${output.length} messages`);
    totalMessages += output.length;
  }

  // Also write the legacy messages.json (challenge #1) for backward compat
  const challengeTopic = toSync.find((t) =>
    t.title.toLowerCase().includes("sparse parity")
  );
  if (challengeTopic) {
    const legacyFile = join(OUTPUT_DIR, "messages.json");
    const slug = slugify(challengeTopic.title);
    const sourceFile = join(OUTPUT_DIR, `${slug}.json`);
    if (existsSync(sourceFile)) {
      const data = await Bun.file(sourceFile).text();
      writeFileSync(legacyFile, data);
    }
  }

  console.log(`\nDone. ${totalMessages} messages across ${toSync.length} topics.`);
  process.exit(0);
}

main().catch((e) => {
  console.error("Error:", e);
  process.exit(1);
});

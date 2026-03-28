import { TelegramClient } from "@mtcute/bun";
import { tl } from "@mtcute/tl";
import { existsSync, writeFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { openDb, upsertTopic, getLastMessageId, insertMessages, exportTopicJson } from "./db";

// --- Config ---
const CHANNEL_USERNAME = "sutro_group";

const TOPICS_TO_SYNC = [
  "chat-yad",
  "chat-yaroslav",
  "challenge #1: sparse parity",
  "General",
  "In-person meetings",
  "Introductions",
];

const SESSION_PATH = join(homedir(), ".telegram-sync-cli", "session_1.db");

function slugify(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

async function fetchTopicMessages(
  client: TelegramClient,
  resolved: tl.TypeInputPeer,
  topicId: number,
  users: Map<number, string>,
  minId: number = 0
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
      minId,
      hash: 0,
    });

    const resp = history as tl.RawMessagesChannelMessages;

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

    await new Promise((r) => setTimeout(r, 500));
  }

  return allMessages;
}

export async function sync(options: { exportJson?: boolean; fullSync?: boolean } = {}) {
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

  const db = openDb();

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

  const users = new Map<number, string>();
  if ("users" in topics) {
    for (const u of (topics as tl.RawMessagesForumTopics).users) {
      if (u._ === "user") {
        users.set(u.id, [u.firstName, u.lastName].filter(Boolean).join(" "));
      }
    }
  }

  // Match topics
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
    db.close();
    process.exit(1);
  }

  let totalNew = 0;
  let totalExisting = 0;

  for (const topic of toSync) {
    const slug = slugify(topic.title);
    upsertTopic(db, topic.id, topic.title, slug);

    const lastId = options.fullSync ? 0 : getLastMessageId(db, topic.id);
    const mode = lastId > 0 ? `incremental (after msg ${lastId})` : "full backfill";
    console.log(`\nSyncing [${topic.id}] "${topic.title}" (${mode})`);

    const messages = await fetchTopicMessages(client, resolved, topic.id, users, lastId);

    const formatted = messages.map((m) => ({
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

    const newCount = insertMessages(db, topic.id, formatted);
    totalNew += newCount;
    totalExisting += formatted.length - newCount;
    console.log(`  ${formatted.length} fetched, ${newCount} new, ${formatted.length - newCount} already in DB`);
  }

  // Export JSON if requested
  if (options.exportJson) {
    const outDir = join(import.meta.dir, "..", "sparse_parity", "telegram_sync");
    console.log(`\nExporting JSON to ${outDir}`);
    for (const topic of toSync) {
      const slug = slugify(topic.title);
      const json = exportTopicJson(db, topic.id);
      writeFileSync(join(outDir, `${slug}.json`), json);
    }
    // Legacy messages.json
    const challengeTopic = toSync.find((t) =>
      t.title.toLowerCase().includes("sparse parity")
    );
    if (challengeTopic) {
      const json = exportTopicJson(db, challengeTopic.id);
      writeFileSync(join(outDir, "messages.json"), json);
    }
    console.log("  JSON export complete.");
  }

  db.close();
  console.log(`\nDone. ${totalNew} new messages, ${totalExisting} already synced.`);
  process.exit(0);
}

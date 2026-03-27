#!/usr/bin/env bun
// Read messages from a Telegram forum topic.
// Usage: bun telegram/tg-read.ts --topic "General" --limit 20
//        bun telegram/tg-read.ts --topic 123 --limit 50 --since 2025-01-01
// Output: JSON array of { id, date, sender, text, replyTo }

import { tl } from "@mtcute/tl";
import { parseArgs } from "util";
import { getClient } from "./client.ts";
import { resolveTopic } from "./resolve-topic.ts";

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    topic: { type: "string" },
    limit: { type: "string", default: "50" },
    since: { type: "string" },
  },
});

if (!values.topic) {
  console.error("Usage: bun tg-read.ts --topic <name|id> [--limit N] [--since ISO-date]");
  process.exit(1);
}

const limit = parseInt(values.limit!, 10);
const sinceDate = values.since ? new Date(values.since).getTime() / 1000 : 0;

const { client, peer, inputChannel } = await getClient();
const topic = await resolveTopic(client, inputChannel, values.topic);

// Fetch messages with pagination (adapted from sync_telegram.ts)
const allMessages: tl.RawMessage[] = [];
const users = new Map<number, string>();
let offsetId = 0;
const BATCH_SIZE = 100;

while (allMessages.length < limit) {
  const history = await client.call({
    _: "messages.getReplies",
    peer,
    msgId: topic.id,
    offsetId,
    offsetDate: 0,
    addOffset: 0,
    limit: Math.min(BATCH_SIZE, limit - allMessages.length),
    maxId: 0,
    minId: 0,
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

  for (const m of msgs) {
    if (sinceDate && m.date < sinceDate) {
      // Messages are newest-first; once we pass the cutoff, stop
      break;
    }
    allMessages.push(m);
    if (allMessages.length >= limit) break;
  }

  // If we hit the since cutoff, stop paginating
  if (sinceDate && msgs[msgs.length - 1].date < sinceDate) break;

  offsetId = msgs[msgs.length - 1].id;
  await new Promise((r) => setTimeout(r, 500));
}

const output = allMessages.map((m) => ({
  id: m.id,
  date: new Date(m.date * 1000).toISOString(),
  sender:
    users.get(
      Number(m.fromId && "userId" in m.fromId ? m.fromId.userId : 0)
    ) ?? String(m.fromId),
  text: m.message,
  replyTo:
    m.replyTo && "replyToMsgId" in m.replyTo ? m.replyTo.replyToMsgId : null,
}));

console.log(JSON.stringify(output, null, 2));
process.exit(0);

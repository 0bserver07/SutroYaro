#!/usr/bin/env bun
// Send a message to a Telegram forum topic.
// Usage: bun telegram/tg-send.ts --topic "agents" --message "Hello from agent"
//        echo "Multi-line msg" | bun telegram/tg-send.ts --topic "agents" --stdin
//        bun telegram/tg-send.ts --message "Hello"   # uses TELEGRAM_WRITE_TOPIC
// Output: JSON { ok, topicId, topicTitle }

import { parseArgs } from "util";
import { getClient } from "./client.ts";
import { resolveTopic } from "./resolve-topic.ts";
import { WRITE_TOPIC } from "./config.ts";

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    topic: { type: "string" },
    message: { type: "string" },
    stdin: { type: "boolean", default: false },
  },
});

const topicArg = values.topic || WRITE_TOPIC;
if (!topicArg) {
  console.error(
    "Specify --topic <name|id> or set TELEGRAM_WRITE_TOPIC in .env"
  );
  process.exit(1);
}

let message: string;
if (values.stdin) {
  message = await Bun.stdin.text();
} else if (values.message) {
  message = values.message;
} else {
  console.error("Provide --message <text> or --stdin");
  process.exit(1);
}

message = message.trim();
if (!message) {
  console.error("Message is empty");
  process.exit(1);
}

const { client, peer, inputChannel } = await getClient();
const topic = await resolveTopic(client, inputChannel, topicArg);

const sendMessage = async () => {
  await client.call({
    _: "messages.sendMessage",
    peer,
    message,
    replyTo: {
      _: "inputReplyToMessage",
      replyToMsgId: topic.id,
      topMsgId: topic.id,
    },
    randomId: BigInt(crypto.getRandomValues(new BigInt64Array(1))[0]),
  });
};

try {
  await sendMessage();
} catch (e: any) {
  // Handle FLOOD_WAIT: sleep and retry once
  if (e?.errorMessage?.startsWith?.("FLOOD_WAIT_")) {
    const seconds = parseInt(e.errorMessage.split("_").pop(), 10) || 5;
    console.error(`Rate limited, waiting ${seconds}s...`);
    await new Promise((r) => setTimeout(r, seconds * 1000));
    await sendMessage();
  } else {
    throw e;
  }
}

console.log(
  JSON.stringify({ ok: true, topicId: topic.id, topicTitle: topic.title })
);
process.exit(0);

#!/usr/bin/env bun
// List all forum topics in the Telegram group.
// Usage: bun telegram/tg-topics.ts
// Output: JSON array of { id, title }

import { getClient } from "./client.ts";
import { getForumTopics } from "./resolve-topic.ts";

const { client, inputChannel } = await getClient();
const topics = await getForumTopics(client, inputChannel);

const output = topics.map((t) => ({ id: t.id, title: t.title }));
console.log(JSON.stringify(output, null, 2));
process.exit(0);

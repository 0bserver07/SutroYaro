import { TelegramClient } from "@mtcute/bun";
import { tl } from "@mtcute/tl";

export async function getForumTopics(
  client: TelegramClient,
  inputChannel: tl.TypeInputChannel
): Promise<tl.RawForumTopic[]> {
  const topics = await client.call({
    _: "channels.getForumTopics",
    channel: inputChannel,
    limit: 100,
    offsetDate: 0,
    offsetId: 0,
    offsetTopic: 0,
  });

  return (topics as tl.RawMessagesForumTopics).topics.filter(
    (t): t is tl.RawForumTopic => t._ === "forumTopic"
  );
}

export async function resolveTopic(
  client: TelegramClient,
  inputChannel: tl.TypeInputChannel,
  topicArg: string
): Promise<{ id: number; title: string }> {
  // If numeric, use directly
  const asNumber = parseInt(topicArg, 10);
  if (!isNaN(asNumber) && String(asNumber) === topicArg.trim()) {
    // Still fetch topics to get the title
    const topics = await getForumTopics(client, inputChannel);
    const match = topics.find((t) => t.id === asNumber);
    return { id: asNumber, title: match?.title ?? `topic-${asNumber}` };
  }

  // Otherwise, match by case-insensitive substring
  const topics = await getForumTopics(client, inputChannel);
  const match = topics.find((t) =>
    t.title.toLowerCase().includes(topicArg.toLowerCase())
  );

  if (!match) {
    const available = topics.map((t) => `  - ${t.title} (${t.id})`).join("\n");
    console.error(`No topic matching "${topicArg}". Available:\n${available}`);
    process.exit(1);
  }

  return { id: match.id, title: match.title };
}

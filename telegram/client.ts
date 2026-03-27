import { TelegramClient } from "@mtcute/bun";
import { tl } from "@mtcute/tl";
import { existsSync } from "fs";
import { API_ID, API_HASH, SESSION_PATH, CHANNEL_USERNAME } from "./config.ts";

export async function getClient() {
  if (!API_ID || !API_HASH) {
    console.error("Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env");
    process.exit(1);
  }

  if (!existsSync(SESSION_PATH)) {
    console.error(
      `Session not found at ${SESSION_PATH}. Run 'tg auth login' first.`
    );
    process.exit(1);
  }

  const client = new TelegramClient({
    apiId: API_ID,
    apiHash: API_HASH,
    storage: SESSION_PATH,
  });

  await client.start();

  const peer = await client.resolvePeer(CHANNEL_USERNAME);
  const inputChannel: tl.TypeInputChannel = {
    _: "inputChannel",
    channelId: (peer as tl.RawInputPeerChannel).channelId,
    accessHash: (peer as tl.RawInputPeerChannel).accessHash,
  };

  return { client, peer, inputChannel };
}

import { join } from "path";
import { homedir } from "os";

export const CHANNEL_USERNAME = process.env.TELEGRAM_CHANNEL ?? "sutro_group";
export const SESSION_PATH =
  process.env.TELEGRAM_SESSION ??
  join(homedir(), ".telegram-sync-cli", "session_1.db");
export const API_ID = parseInt(process.env.TELEGRAM_API_ID ?? "0", 10);
export const API_HASH = process.env.TELEGRAM_API_HASH ?? "";
export const WRITE_TOPIC = process.env.TELEGRAM_WRITE_TOPIC ?? "";

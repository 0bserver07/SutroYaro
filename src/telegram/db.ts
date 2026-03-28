import { Database } from "bun:sqlite";
import { join } from "path";

const DB_PATH = join(import.meta.dir, "..", "..", "telegram.db");

export function openDb(): Database {
  const db = new Database(DB_PATH);
  db.run("PRAGMA journal_mode = WAL");
  db.run("PRAGMA foreign_keys = ON");

  db.run(`
    CREATE TABLE IF NOT EXISTS topics (
      id INTEGER PRIMARY KEY,
      title TEXT NOT NULL,
      slug TEXT NOT NULL UNIQUE
    )
  `);

  db.run(`
    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER NOT NULL,
      topic_id INTEGER NOT NULL,
      date TEXT NOT NULL,
      sender TEXT NOT NULL,
      text TEXT NOT NULL DEFAULT '',
      reply_to INTEGER,
      PRIMARY KEY (id, topic_id),
      FOREIGN KEY (topic_id) REFERENCES topics(id)
    )
  `);

  db.run(`
    CREATE INDEX IF NOT EXISTS idx_messages_topic_date
    ON messages(topic_id, date DESC)
  `);

  db.run(`
    CREATE INDEX IF NOT EXISTS idx_messages_sender
    ON messages(sender)
  `);

  // Track sync state per topic
  db.run(`
    CREATE TABLE IF NOT EXISTS sync_state (
      topic_id INTEGER PRIMARY KEY,
      last_message_id INTEGER NOT NULL DEFAULT 0,
      last_sync TEXT NOT NULL,
      FOREIGN KEY (topic_id) REFERENCES topics(id)
    )
  `);

  return db;
}

export function upsertTopic(db: Database, id: number, title: string, slug: string) {
  db.run(
    "INSERT INTO topics (id, title, slug) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET title = excluded.title, slug = excluded.slug",
    [id, title, slug]
  );
}

export function getLastMessageId(db: Database, topicId: number): number {
  const row = db.query("SELECT last_message_id FROM sync_state WHERE topic_id = ?").get(topicId) as { last_message_id: number } | null;
  return row?.last_message_id ?? 0;
}

export function insertMessages(
  db: Database,
  topicId: number,
  messages: { id: number; date: string; sender: string; text: string; replyTo: number | null }[]
) {
  if (messages.length === 0) return 0;

  const insert = db.prepare(
    "INSERT OR IGNORE INTO messages (id, topic_id, date, sender, text, reply_to) VALUES (?, ?, ?, ?, ?, ?)"
  );

  const tx = db.transaction(() => {
    let count = 0;
    for (const m of messages) {
      const result = insert.run(m.id, topicId, m.date, m.sender, m.text, m.replyTo);
      if (result.changes > 0) count++;
    }

    // Update sync state
    const maxId = Math.max(...messages.map((m) => m.id));
    db.run(
      "INSERT INTO sync_state (topic_id, last_message_id, last_sync) VALUES (?, ?, ?) ON CONFLICT(topic_id) DO UPDATE SET last_message_id = MAX(excluded.last_message_id, sync_state.last_message_id), last_sync = excluded.last_sync",
      [topicId, maxId, new Date().toISOString()]
    );

    return count;
  });

  return tx();
}

export function exportTopicJson(db: Database, topicId: number): string {
  const rows = db.query(
    "SELECT id, date, sender, text, reply_to as replyTo FROM messages WHERE topic_id = ? ORDER BY date DESC"
  ).all(topicId);
  return JSON.stringify(rows, null, 2);
}

export { DB_PATH };

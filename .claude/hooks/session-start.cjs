#!/usr/bin/env node
/**
 * Session-start hook: shows project status when a Claude Code session begins.
 * Outputs JSON with systemMessage for Claude Code hook protocol.
 */

const { execSync } = require("child_process");

function run(cmd) {
  try {
    return execSync(cmd, { encoding: "utf8", timeout: 5000 }).trim();
  } catch {
    return null;
  }
}

function main() {
  const lines = [];
  lines.push("--- SutroYaro Session Status ---");

  const branch = run("git branch --show-current");
  const diffStat = run("git diff --shortstat");
  const lastCommit = run('git log --oneline -1 --format="%s"');
  if (branch) {
    let gitLine = `Branch: ${branch}`;
    if (diffStat) gitLine += ` (${diffStat})`;
    lines.push(gitLine);
  }
  if (lastCommit) {
    lines.push(`Last commit: ${lastCommit}`);
  }

  const issueCount = run(
    "gh issue list --repo cybertronai/SutroYaro --state open --json number --jq length 2>/dev/null"
  );
  if (issueCount) {
    lines.push(`Open issues: ${issueCount}`);
  }

  const lastExp = run(
    'tail -1 research/log.jsonl 2>/dev/null | python3 -c "import sys,json; e=json.load(sys.stdin); print(f\\"{e.get(\'id\',\'?\')}: {e.get(\'result\',\'?\')}\\")" 2>/dev/null'
  );
  if (lastExp) {
    lines.push(`Last experiment: ${lastExp}`);
  }

  const todoCount = run(
    'grep -c "TODO\\|IN PROGRESS" docs/tasks/INDEX.md 2>/dev/null'
  );
  if (todoCount && parseInt(todoCount) > 0) {
    lines.push(`Open tasks: ${todoCount}`);
  }

  const telegramAge = run(
    'stat -f "%Sm" -t "%Y-%m-%d" src/sparse_parity/telegram_sync/messages.json 2>/dev/null'
  );
  if (telegramAge) {
    lines.push(`Telegram sync: ${telegramAge}`);
  }

  const gdocsAge = run(
    'stat -f "%Sm" -t "%Y-%m-%d" docs/google-docs/sutro-group-main.md 2>/dev/null'
  );
  if (gdocsAge) {
    lines.push(`Google Docs sync: ${gdocsAge}`);
  }

  lines.push("---");

  const result = {
    continue: true,
    systemMessage: lines.join("\n"),
  };
  console.log(JSON.stringify(result));
  process.exit(0);
}

main();

#!/usr/bin/env node
/**
 * Session-start hook: shows project status when a Claude Code session begins.
 */

const common = require("./hook-common.cjs");

function main() {
  const cwd = process.cwd();
  const lines = ["--- SutroYaro Session Status ---"];

  const git = common.getGitInfo(cwd);
  if (git.is_repo) {
    let gitLine = `Branch: ${git.branch}`;
    if (git.diff) gitLine += ` (${git.diff})`;
    lines.push(gitLine);
    if (git.last_commit) lines.push(`Last commit: ${git.last_commit}`);
  }

  const issueCount = common.run(
    "gh issue list --repo cybertronai/SutroYaro --state open --json number --jq length 2>/dev/null",
    cwd
  );
  if (issueCount) lines.push(`Open issues: ${issueCount}`);

  const lastExp = common.getLastExperiment(cwd);
  if (lastExp) lines.push(`Last experiment: ${lastExp}`);

  const tasks = common.getOpenTaskCount(cwd);
  if (tasks > 0) lines.push(`Open tasks: ${tasks}`);

  const sync = common.getSyncStatus(cwd);
  if (sync.telegram) lines.push(`Telegram sync: ${sync.telegram}`);
  if (sync.gdocs) lines.push(`Google Docs sync: ${sync.gdocs}`);

  lines.push("---");

  console.log(JSON.stringify({ continue: true, systemMessage: lines.join("\n") }));
  process.exit(0);
}

main();

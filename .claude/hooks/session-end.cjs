#!/usr/bin/env node
/**
 * Session-end hook: summarizes what changed during the session.
 * Outputs JSON per Claude Code hook protocol.
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
  lines.push("--- Session Summary ---");

  const uncommitted = run("git diff --shortstat");
  const staged = run("git diff --cached --shortstat");
  if (uncommitted) lines.push(`Uncommitted: ${uncommitted}`);
  if (staged) lines.push(`Staged: ${staged}`);

  const recentCommits = run('git log --oneline -3 --format="%h %s"');
  if (recentCommits) {
    lines.push("Recent commits:");
    recentCommits.split("\n").forEach((c) => lines.push(`  ${c}`));
  }

  const indexMtime = run(
    'stat -f "%Sm" -t "%s" docs/tasks/INDEX.md 2>/dev/null'
  );
  if (indexMtime) {
    const ageHours = (Date.now() / 1000 - parseInt(indexMtime)) / 3600;
    if (ageHours > 24) {
      lines.push(
        `Tasks INDEX.md last updated ${Math.floor(ageHours / 24)} days ago`
      );
    }
  }

  const unpushed = run(
    "git log --oneline @{upstream}..HEAD 2>/dev/null | wc -l"
  );
  if (unpushed && parseInt(unpushed.trim()) > 0) {
    lines.push(`Unpushed commits: ${unpushed.trim()}`);
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

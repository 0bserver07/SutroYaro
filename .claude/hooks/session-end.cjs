#!/usr/bin/env node
/**
 * Session-end hook: summarizes what changed during the session.
 */

const common = require("./hook-common.cjs");
const fs = require("fs");
const path = require("path");

function main() {
  const cwd = process.cwd();
  const lines = ["--- Session Summary ---"];

  const git = common.getGitInfo(cwd);
  if (git.diff) lines.push(`Uncommitted: ${git.diff}`);

  const staged = common.run("git diff --cached --shortstat", cwd);
  if (staged) lines.push(`Staged: ${staged}`);

  const recentCommits = common.run('git log --oneline -3 --format="%h %s"', cwd);
  if (recentCommits) {
    lines.push("Recent commits:");
    recentCommits.split("\n").forEach((c) => lines.push(`  ${c}`));
  }

  const indexPath = path.join(cwd, "docs", "tasks", "INDEX.md");
  if (fs.existsSync(indexPath)) {
    const stat = fs.statSync(indexPath);
    const ageHours = (Date.now() - stat.mtimeMs) / 3600000;
    if (ageHours > 24) {
      lines.push(`Tasks INDEX.md last updated ${Math.floor(ageHours / 24)} days ago`);
    }
  }

  const unpushed = common.run("git log --oneline @{upstream}..HEAD 2>/dev/null | wc -l", cwd);
  if (unpushed && parseInt(unpushed.trim()) > 0) {
    lines.push(`Unpushed commits: ${unpushed.trim()}`);
  }

  lines.push("---");

  console.log(JSON.stringify({ continue: true, systemMessage: lines.join("\n") }));
  process.exit(0);
}

main();

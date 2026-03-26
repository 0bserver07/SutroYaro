/**
 * Shared utility library for SutroYaro hooks.
 * Keep this small -- only extract what multiple hooks actually use.
 */

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

function run(cmd, cwd) {
  try {
    return execSync(cmd, { cwd, encoding: "utf8", timeout: 5000, stdio: "pipe" }).trim();
  } catch {
    return null;
  }
}

function getGitInfo(cwd) {
  const branch = run("git branch --show-current", cwd);
  if (!branch) return { is_repo: false, branch: null, diff: null, last_commit: null };

  return {
    is_repo: true,
    branch,
    diff: run("git diff --shortstat", cwd),
    last_commit: run('git log --oneline -1 --format="%s"', cwd),
  };
}

function getSyncStatus(cwd) {
  const result = {};

  const telegramPath = path.join(cwd, "src", "sparse_parity", "telegram_sync", "messages.json");
  if (fs.existsSync(telegramPath)) {
    const stat = fs.statSync(telegramPath);
    result.telegram = stat.mtime.toISOString().slice(0, 10);
  }

  const gdocsPath = path.join(cwd, "docs", "google-docs", "sutro-group-main.md");
  if (fs.existsSync(gdocsPath)) {
    const stat = fs.statSync(gdocsPath);
    result.gdocs = stat.mtime.toISOString().slice(0, 10);
  }

  return result;
}

function getOpenTaskCount(cwd) {
  const indexPath = path.join(cwd, "docs", "tasks", "INDEX.md");
  if (!fs.existsSync(indexPath)) return 0;
  const content = fs.readFileSync(indexPath, "utf8");
  const matches = content.match(/TODO|IN PROGRESS/g);
  return matches ? matches.length : 0;
}

function getLastExperiment(cwd) {
  const logPath = path.join(cwd, "research", "log.jsonl");
  if (!fs.existsSync(logPath)) return null;
  const last = run(`tail -1 "${logPath}"`, cwd);
  if (!last) return null;
  try {
    const e = JSON.parse(last);
    const r = e.result || {};
    const acc = r.accuracy != null ? `acc=${r.accuracy}` : "";
    const cls = e.class || "";
    return `${e.id || "?"} (${e.method || "?"}): ${cls} ${acc}`.trim();
  } catch {
    return null;
  }
}

module.exports = {
  run,
  getGitInfo,
  getSyncStatus,
  getOpenTaskCount,
  getLastExperiment,
};

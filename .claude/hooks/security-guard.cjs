#!/usr/bin/env node
/**
 * Security guard hook: blocks modifications to locked files and destructive commands.
 * Two-tier system: Tier 1 (deny) blocks immediately, Tier 2 (confirm) asks user.
 * Outputs JSON per Claude Code hook protocol.
 */

// Measurement code that should not be edited during experiments (LAB.md rule #9).
// harness.py is NOT locked here because it needs legitimate edits when adding
// new methods. Rule #9 enforcement for harness.py relies on LAB.md and PR review.
//
// Future: a smarter approach could check if the current branch is an experiment
// branch (exp-*) vs a development branch, and only lock files on experiment branches.
const LOCKED_FILES = [
  "src/sparse_parity/tracker.py",
  "src/sparse_parity/cache_tracker.py",
  "src/sparse_parity/data.py",
  "src/sparse_parity/config.py",
];

// Tier 1: block immediately
const DESTRUCTIVE_PATTERNS = [
  { pattern: /rm\s+-rf\s+(?!\/tmp)/, label: "rm -rf outside /tmp" },
  { pattern: /rm\s+-r\s+(?!\/tmp)/, label: "rm -r outside /tmp" },
  { pattern: /git\s+reset\s+--hard/, label: "git reset --hard" },
  { pattern: /git\s+clean\s+-f/, label: "git clean -f" },
];

// Tier 2: ask for confirmation
const CONFIRM_PATTERNS = [
  { pattern: /git\s+push\s+--force/, label: "git push --force" },
  { pattern: /git\s+push\s+-f\b/, label: "git push -f" },
  { pattern: /git\s+branch\s+-D/, label: "git branch -D (force delete)" },
];

const PROTECTED_DOTFILES = [
  ".zshrc", ".bashrc", ".zsh_history", ".gitconfig", ".bash_profile",
];

function main() {
  let inputData = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk) => {
    inputData += chunk;
  });
  process.stdin.on("end", () => {
    try {
      const event = JSON.parse(inputData);
      const toolName = event.tool_name || "";
      const input = event.tool_input || {};

      let decision = "allow";
      let reason = "";
      let confirmLabel = "";

      if (toolName === "Bash") {
        const command = input.command || "";

        // Tier 1: deny
        for (const { pattern, label } of DESTRUCTIVE_PATTERNS) {
          if (pattern.test(command)) {
            decision = "deny";
            reason = `Destructive command: ${label}`;
            break;
          }
        }

        // Tier 2: confirm
        if (decision === "allow") {
          for (const { pattern, label } of CONFIRM_PATTERNS) {
            if (pattern.test(command)) {
              confirmLabel = label;
              break;
            }
          }
        }

        // Dotfile protection
        if (decision === "allow") {
          for (const dotfile of PROTECTED_DOTFILES) {
            if (command.includes(dotfile) && (command.includes(">") || command.includes("sed") || command.includes("echo"))) {
              decision = "deny";
              reason = `Dotfile modification: ${dotfile}`;
              break;
            }
          }
        }
      } else if (toolName === "Edit" || toolName === "Write") {
        const filePath = input.file_path || "";
        for (const locked of LOCKED_FILES) {
          if (filePath.endsWith(locked)) {
            decision = "deny";
            reason = `Locked file (LAB.md rule #9): ${locked}`;
            break;
          }
        }
      }

      // Output per Claude Code hook protocol
      // See: https://code.claude.com/docs/en/hooks
      if (decision === "deny") {
        const errorOutput = {
          hookSpecificOutput: {
            hookEventName: "PreToolUse",
            permissionDecision: "deny",
            permissionDecisionReason: reason,
          },
          systemMessage: `BLOCKED: ${reason}\n\nTo perform this operation, run it manually in the terminal.`,
        };
        console.error(JSON.stringify(errorOutput));
        process.exit(2);
      } else {
        const result = {};
        if (confirmLabel) {
          result.hookSpecificOutput = {
            hookEventName: "PreToolUse",
            permissionDecision: "ask",
            permissionDecisionReason: confirmLabel,
          };
          result.systemMessage = `CONFIRM REQUIRED: ${confirmLabel}\n\nYou MUST ask the user for explicit confirmation before executing this operation.`;
        }
        console.log(JSON.stringify(result));
        process.exit(0);
      }
    } catch {
      // Can't parse input, don't block
      console.log(JSON.stringify({ continue: true }));
      process.exit(0);
    }
  });
}

main();

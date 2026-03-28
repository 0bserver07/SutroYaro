# Agent CLI Setup Guide

How to set up and run autonomous experiments with each supported AI tool.

All tools use the same protocol (AGENT.md), the same locked harness (src/harness.py), and the same log format (research/log.jsonl). The tool doesn't matter -- the protocol does.

## Prerequisites (all tools)

```bash
git clone https://github.com/cybertronai/SutroYaro.git
cd SutroYaro

# Verify environment
PYTHONPATH=src python3 checks/env_check.py

# Establish baselines on your machine
PYTHONPATH=src python3 checks/baseline_check.py
```

Both checks must pass before running experiments. If they fail, fix the issue before proceeding.

---

## Claude Code

Claude Code is a terminal-native AI agent from Anthropic. It has file read/write, bash execution, and tool permissions built in.

### Install

```bash
# macOS / Linux
npm install -g @anthropic-ai/claude-code

# Verify
claude --version
```

You need an Anthropic API key or a Claude Pro/Team subscription.

### Run experiments

```bash
# Single cycle, 10 experiments
bin/run-agent --tool claude --max 10

# Overnight: 10 cycles, 5 experiments each
bin/run-agent --tool claude --loop 10 --max 5

# With researcher ID
bin/run-agent --tool claude --max 10 --researcher yad-agent
```

### How it launches

The launcher calls `claude -p "$prompt" --allowedTools "Bash,Read,Write,Edit,Glob,Grep" --max-turns 200`. The prompt tells the agent to read AGENT.md and follow its protocol.

### Customization

Claude Code reads `CLAUDE.md` at the project root automatically. This file already contains the project context, rules, and file references. You can also use:

- **Skills/plugins**: custom slash commands (e.g., `/commit`, analysis tools)
- **MCP servers**: extend with external tools (calendar, GitHub, etc.)
- **Hooks**: scripts that run on tool calls (stop hooks, pre-commit checks)
- **AGENTS.md**: sub-agent dispatch instructions

---

## Gemini CLI

Gemini CLI is Google's open-source terminal AI agent. It has a 1M token context window, MCP server support, and a free tier.

### Install

```bash
npm install -g @google/gemini-cli

# Verify
gemini --version
```

Free tier with a personal Google account: 60 requests/min, 1,000 requests/day. Or use a Google AI Studio / Vertex AI key for more.

### Run experiments

```bash
# Single cycle
bin/run-agent --tool gemini --max 10

# Overnight
bin/run-agent --tool gemini --loop 10 --max 5
```

### How it launches

The launcher calls `gemini -p "$prompt" --yolo`. The `-p` flag runs in headless mode (no interactive UI). `--yolo` skips confirmation prompts so the agent can run bash commands without pausing.

### Customization

Gemini CLI reads `GEMINI.md` at the project root if it exists. You could create one that mirrors CLAUDE.md, or just rely on AGENT.md (which the prompt tells the agent to read).

- **MCP servers**: same protocol as Claude Code, connect external tools
- **Extensions**: custom commands via `.gemini/` config directory
- **TOML config**: per-project settings in `.gemini/settings.toml`

### Gotcha

Gemini CLI's custom slash commands don't work in headless mode yet (known issue). The agent can still read files and run bash, which is all our protocol needs.

---

## Antigravity

Google Antigravity is an agent-first IDE (like VS Code but built around AI agents). It has a Manager View for dispatching parallel agents.

### Install

Download from [antigravity.google](https://antigravity.google) or via the command line tool:

```bash
# The CLI command is 'agy'
agy .    # opens the current directory in Antigravity
```

Free for individuals in public preview.

### Run experiments (manual workflow)

Antigravity is an IDE, not a headless CLI. It cannot be driven by `bin/run-agent`. Instead:

1. Open the project: `agy .`
2. In the agent chat, type: "Read AGENT.md. Follow its protocol exactly. Your researcher ID is [your-name]."
3. The agent will read the files, pick hypotheses from TODO.md, run experiments, and log results.

### Parallel agents via Manager View

Antigravity's unique feature is Manager View, where you can dispatch multiple agents in parallel:

1. Open Manager View
2. Create Agent 1: "Read AGENT.md. Test the first unchecked hypothesis in TODO.md."
3. Create Agent 2: "Read AGENT.md. Test the second unchecked hypothesis in TODO.md."
4. Both agents work simultaneously on different hypotheses

This is similar to Claude Code's sub-agent dispatch but with a visual interface.

### Customization

- **Multi-model**: supports Claude Sonnet/Opus, Gemini, and GPT models
- **Artifacts**: agents produce visible task lists, plans, and screenshots
- **No MCP yet**: MCP server support is not available as of early 2026

---

## Codex CLI

OpenAI's terminal coding agent, powered by codex-1 (optimized o3). Built in Rust, runs locally.

### Install

```bash
npm install -g @openai/codex
# or
brew install --cask codex

# Verify
codex --version
```

### Authentication (OAuth / ChatGPT subscription)

Codex CLI supports two auth methods. **OAuth login is recommended** because it uses your existing ChatGPT subscription (Plus, Pro, Business, Edu, Enterprise) with no separate API billing.

**Option 1: ChatGPT OAuth (recommended -- uses subscription, no API credits)**

```bash
codex --login
```

This opens a browser window for ChatGPT OAuth. Sign in with your ChatGPT account. The session token is stored locally. No API key needed, no usage charges beyond your subscription.

The project's `.codex/config.toml` already sets `forced_login_method = "chatgpt"` so Codex will prefer OAuth by default in this repo.

**Option 2: API key (pay-per-token)**

```bash
export OPENAI_API_KEY="sk-..."
```

Set the key in your shell profile or `.env`. This bills per token against your OpenAI API account. Use this if you don't have a ChatGPT subscription or need higher rate limits.

**Verifying auth**:

```bash
codex --version   # should show version without auth errors
codex "echo hello" # quick smoke test
```

### Run experiments

```bash
# Single cycle
bin/run-agent --tool codex --max 10

# Overnight
bin/run-agent --tool codex --loop 10 --max 5
```

### How it launches

The launcher calls `codex -q "$prompt"`. The `-q` flag runs in quiet/non-interactive mode.

### Project config

This repo includes Codex-specific configuration for feature parity with Claude Code:

| File | Purpose | Claude Code equivalent |
|------|---------|----------------------|
| `.codex/config.toml` | Sandbox mode, approval policy, model, auth | `.claude/settings.local.json` |
| `CODEX.md` | Project context (auto-loaded at session start) | `CLAUDE.md` |
| `.codex/AGENTS.md` | Agent instructions: context loading, sync, writing rules | `.claude/skills/` |

Codex auto-reads `CODEX.md` and `.codex/AGENTS.md` at session start, giving it the same project context, sync routines, and anti-slop writing rules that Claude Code gets from `CLAUDE.md` and `.claude/skills/`.

### Customization

- **CODEX.md**: project context loaded at session start (same content as CLAUDE.md)
- **.codex/AGENTS.md**: instructions for context loading, sync routine, writing rules
- **.codex/config.toml**: sandbox mode, approval policy, model selection, auth method
- **MCP servers**: supported for external tool integration
- **Multi-agent**: experimental parallel agent support built in
- **Sandbox**: cross-platform security (macOS Seatbelt, Linux Landlock)

---

## OpenCode

Open-source, multi-provider terminal agent. Works with Claude, GPT, Gemini, and local models (Ollama). 95K+ GitHub stars. No vendor lock-in.

### Install

```bash
curl -fsSL https://opencode.ai/install | bash
# or
npm install -g opencode-ai@latest
# or
brew install opencode

# Verify
opencode --version
```

Bring your own API key (any provider). Or use local models for free.

### Run experiments

```bash
# Single cycle
bin/run-agent --tool opencode --max 10

# Overnight
bin/run-agent --tool opencode --loop 10 --max 5
```

### How it launches

OpenCode accepts prompts as command-line arguments for non-interactive mode. The launcher passes the prompt directly.

### Customization

- **Multi-provider**: switch models mid-session (Claude for reasoning, GPT for code generation, local for privacy)
- **LSP integration**: type information and symbol definitions from your language server
- **Session persistence**: SQLite-backed session storage
- **No vendor lock-in**: the main reason to choose OpenCode over single-vendor CLIs

---

## Custom CLI

Any AI tool that can accept a prompt and execute commands can be used.

```bash
# Set AI_CMD to your tool's command
AI_CMD="my-ai-tool --prompt" bin/run-agent --tool custom --max 10
```

The launcher pipes the prompt to `$AI_CMD` via stdin. Your tool needs to:

1. Accept a text prompt
2. Be able to read/write files in the working directory
3. Be able to run `python3` commands
4. Write output to stdout (for logging)

---

## After any run

```bash
# Text report
bin/analyze-log

# Generate progress chart
bin/analyze-log --plot

# Merge your results for PR submission
bin/merge-findings research/log.jsonl --scoreboard
```

## Comparing tools

| Feature | Claude Code | Gemini CLI | Codex CLI | OpenCode | Antigravity |
|---------|------------|-----------|-----------|----------|-------------|
| Headless mode | Yes (`-p`) | Yes (`-p --yolo`) | Yes (`-q`) | Yes (arg) | No (IDE) |
| `bin/run-agent` | Yes | Yes | Yes | Yes | No |
| Looped overnight | Yes | Yes | Yes | Yes | No |
| MCP servers | Yes | Yes | Yes | Yes | No |
| Custom skills/plugins | Yes | Yes (extensions) | Yes | Yes | No |
| Project context file | `CLAUDE.md` | `GEMINI.md` | `CODEX.md` | -- | -- |
| Agent instructions | `.claude/skills/` | `.gemini/` | `.codex/AGENTS.md` | -- | -- |
| OAuth (subscription) | Yes | Yes (Google) | Yes (ChatGPT) | No | Yes (Google) |
| Multi-provider | No (Anthropic) | No (Google) | No (OpenAI) | Yes (75+) | Yes |
| Context window | 200K (Opus: 1M) | 1M | Varies | Varies | Varies |
| Cost | Subscription or API | Free tier | ChatGPT Plus or API | BYO API key | Free preview |
| Open source | No | Yes (Apache 2.0) | Yes | Yes | No |

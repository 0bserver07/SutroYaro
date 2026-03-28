# What's New (March 2026)

For the Sutro Group. If you want to run experiments on sparse parity (or the next challenge) using your own AI tool, this page tells you how.

## What we added

Before: each person ran experiments with their own tool, measured things differently, and results lived in scattered docs. Without a shared measurement standard, it was hard to compare results across people and tools.

Now: everyone runs against the same locked harness. The harness measures ARD, DMC, and wall-clock time identically for every tool. Results go into one shared log. Numbers are directly comparable regardless of who ran them or what tool they used.

The system has five parts:

- **`src/harness.py`** -- locked evaluation. Runs GF(2), SGD, KM, Fourier, or SMT and returns accuracy, ARD, DMC, timing. Agents cannot modify this file.
- **`AGENT.md`** -- the protocol. Any AI tool reads this and knows what to do: pick a hypothesis, run it, classify the result (WIN/LOSS/INVALID), log it, repeat.
- **`bin/run-agent`** -- a bash launcher that works with Claude Code, Gemini CLI, Codex CLI, OpenCode, or any other CLI. No hooks, no special setup.
- **`research/log.jsonl`** -- machine-readable log of all 36 experiments so far. One JSON line per experiment.
- **`bin/analyze-log`** -- prints a progress report and generates a chart.

## Get started

```bash
# First time
git clone https://github.com/cybertronai/SutroYaro.git
cd SutroYaro

# Already have the repo
cd SutroYaro
git pull

# Check your environment (needs Python 3.8+ and numpy)
PYTHONPATH=src python3 checks/env_check.py
PYTHONPATH=src python3 checks/baseline_check.py

# See what 36 experiments look like
bin/analyze-log
```

## Run experiments with your tool

Pick whichever AI CLI you already have installed:

```bash
bin/run-agent --tool claude --max 5       # Claude Code
bin/run-agent --tool gemini --max 5       # Gemini CLI
bin/run-agent --tool codex --max 5        # Codex CLI
bin/run-agent --tool opencode --max 5     # OpenCode
```

For Antigravity (which is an IDE, not a CLI):

```bash
agy .
# Then tell the agent: "Read AGENT.md. Follow its protocol."
```

For overnight runs, looped mode runs multiple short cycles. If one crashes, the next picks up from the file state:

```bash
bin/run-agent --tool gemini --loop 10 --max 5    # 10 cycles, up to 5 experiments each
```

Install links if you don't have one yet:

| Tool | Install |
|------|---------|
| Claude Code | `npm i -g @anthropic-ai/claude-code` |
| Gemini CLI | `npm i -g @google/gemini-cli` |
| Codex CLI | `npm i -g @openai/codex` |
| OpenCode | `brew install opencode` |
| Antigravity | [antigravity.google](https://antigravity.google) |

Full setup and customization options per tool: [Agent CLI Guide](../tooling/agent-cli-guide.md)

## Share your results

After running experiments, your results are in `research/log.jsonl`. To merge them into the shared log:

1. Fork the repo (or create a branch)
2. Run your experiments
3. Submit a PR with your updated `log.jsonl` and any findings docs
4. `bin/merge-findings` deduplicates and integrates

The locked harness is what makes this work. Everyone measures the same way, so Yad's Claude Code results are directly comparable to Yaroslav's Gemini results.

## Why we built it this way

The short version: research is about finding the right experiment to run, not just running experiments faster. A coding agent (Claude Code, Gemini CLI, Codex, etc.) can read what's been tried, pick the next hypothesis, run it, log the result, and repeat. That loop is what makes autonomous research possible.

The longer version, with examples from our 36 experiments: [Research as Navigation](navigation-thesis.md)

One concrete example: we ran 4 local learning rules (Hebbian, Predictive Coding, Equilibrium Propagation, Target Propagation). All failed for the same reason -- parity is invisible to methods limited to local statistics. A smarter navigation protocol would have tested 1, understood why it failed, and skipped the other 3. That's the difference between running experiments and navigating a research space.

## Status

What's working:
- Locked harness, all 5 methods verified
- Pre-flight checks (env + baselines)
- Experiment log with all 36 experiments
- Progress report and chart generation
- Tool-agnostic launcher (Claude Code and Gemini CLI tested)
- Merge workflow for cross-researcher results

Work in progress:
- End-to-end test of a full autonomous cycle (harness and launcher work individually, full loop not yet run overnight)
- Codex CLI and OpenCode integration (written from docs, not tested locally yet)
- nanoGPT as the next challenge (protocol supports it, harness doesn't yet)

## All the docs

| Page | What it covers |
|------|---------------|
| [Agent CLI Guide](../tooling/agent-cli-guide.md) | Setup, install, customization for each AI tool |
| [Peer Research Protocol](peer-research-protocol.md) | Full design: two-layer architecture, log schema, nanoGPT migration |
| [Research as Navigation](navigation-thesis.md) | The thesis: research is navigation, coding agents are the right tool (ELI5 through PhD) |
| [Practitioner's Field Guide](survey.md) | All 36 experiments ranked with methodology |
| [AGENT.md](https://github.com/cybertronai/SutroYaro/blob/main/AGENT.md) | The protocol any AI tool follows |
| [DISCOVERIES.md](https://github.com/cybertronai/SutroYaro/blob/main/DISCOVERIES.md) | Every proven fact from 36 experiments |
| [CONTRIBUTING.md](https://github.com/cybertronai/SutroYaro/blob/main/CONTRIBUTING.md) | How to submit your results via PR |

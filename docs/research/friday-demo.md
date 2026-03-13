# Friday Demo

Walk-through for the agentic research session. Five steps, about 15 minutes.

## 1. The problem we solved

Everyone in the group uses different AI tools. Without a shared measurement standard, results aren't comparable. Someone's "ARD improved 30%" means nothing if they measured differently.

We built a locked evaluation harness that everyone shares. Same code measures the same thing regardless of who runs it or what tool they use.

Try it:

```bash
# This runs GF(2) Gaussian elimination on 20-bit sparse parity
# and returns accuracy, ARD, DMC, and timing as JSON
PYTHONPATH=src python3 src/harness.py --method gf2 --n_bits 20 --k_sparse 3 --json
```

You should see something like:

```json
{
  "accuracy": 1.0,
  "ard": 420.0,
  "dmc": 8607.4,
  "time_s": 0.048,
  "method": "gf2"
}
```

Those numbers are the same on every machine. That's the point.

The harness supports 5 methods: `gf2`, `sgd`, `km`, `fourier`, `smt`. Agents cannot modify the harness file -- it's locked by rule.

## 2. What 33 experiments look like

We ran 33 experiments over 3 days. All of them are logged in a machine-readable format.

```bash
bin/analyze-log
```

This prints:
- Total experiments and win rate
- Classification breakdown (WIN, LOSS, INCONCLUSIVE, INVALID)
- Which methods were tested and how many times
- Best ARD and best time achieved
- Running-best ARD over time (the three breakthroughs: per-layer SGD, GF(2), RL)

To see the chart:

```bash
bin/analyze-log --plot
open results/progress.png
```

The top chart shows ARD dropping from 17,000 (SGD) to 500 (GF(2)) to 1 (RL). The bottom chart shows how many experiments ran on each date, colored by WIN/LOSS/INCONCLUSIVE.

## 3. Any AI tool, one command

This is the main thing. The same protocol works with whatever tool you already have installed.

```bash
bin/run-agent --help
```

Show the output -- it lists Claude Code, Gemini CLI, Codex CLI, OpenCode, and custom. Then show what a run command looks like:

```bash
# Claude Code
bin/run-agent --tool claude --max 1

# Gemini CLI
bin/run-agent --tool gemini --max 1

# Codex CLI
bin/run-agent --tool codex --max 1

# Antigravity (IDE, not CLI -- open the project and tell the agent what to do)
agy .
# Then: "Read AGENT.md. Follow its protocol."
```

If you want to run one live, pick Claude Code or Gemini CLI (those are tested). Set `--max 1` so it only runs one experiment and finishes quickly. Make sure there's at least one unchecked hypothesis in TODO.md first.

For overnight runs, looped mode runs multiple short cycles. If one crashes, the next picks up:

```bash
bin/run-agent --tool gemini --loop 10 --max 5
```

## 4. What the agent follows

Open AGENT.md and walk through it. The structure:

- **DO NOT STOP** -- the agent keeps going until the queue is empty
- **Before you start** -- read DISCOVERIES.md, check the environment
- **The loop** -- pick hypothesis, design experiment, run against harness, log result, classify (WIN/LOSS/INVALID), repeat
- **What you can change** -- only parameters in search_space.yaml
- **What you cannot change** -- harness.py, tracker.py, data.py (locked files)
- **Circuit breaker** -- stops if 5+ experiments crash in a row

The protocol is the same regardless of tool. The files are the program. The AI is the interpreter.

## 5. How results accumulate across people

Each person runs experiments locally. Results go into `research/log.jsonl`. To share:

```bash
# After running experiments, check your results
bin/analyze-log

# Fork the repo, push your branch, submit a PR
# bin/merge-findings deduplicates when merging
```

The locked harness is what makes cross-researcher comparison work. Yad's Claude Code results use the same ARD formula as Yaroslav's Gemini results.

After merging, the scoreboard updates:

```bash
bin/merge-findings research/log.jsonl --scoreboard
```

## If someone asks "why coding agents?"

Short answer: a coding agent can read what's been tried, run an experiment, log the result, and repeat. A chatbot can suggest experiments but can't run them. A notebook can run them but can't loop. AutoML can loop but can't decide to try a completely different approach.

Longer answer with examples: [Research as Navigation](navigation-thesis.md)

One concrete example from our experiments: we tested 4 local learning rules (Hebbian, Predictive Coding, Equilibrium Propagation, Target Propagation). All 4 failed for the same reason -- parity is invisible to methods that only see local statistics. A smarter agent would have tested 1, understood why it failed, and skipped the other 3. That's the difference between running experiments and navigating a research space.

## Links to share after the demo

| What | Link |
|------|------|
| Start here (group onboarding) | [What's New](whats-new-march-2026.md) |
| Setup your AI tool | [Agent CLI Guide](../tooling/agent-cli-guide.md) |
| Full protocol design | [Peer Research Protocol](peer-research-protocol.md) |
| The thesis (ELI5 to PhD) | [Research as Navigation](navigation-thesis.md) |
| All 33 experiments ranked | [Practitioner's Field Guide](survey.md) |
| GitHub repo | [0bserver07/SutroYaro](https://github.com/0bserver07/SutroYaro) |

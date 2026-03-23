# SutroYaro

Research workspace for the [Sutro Group](https://docs.google.com/document/d/1B9867EN6Bg4ZVQK9vI_ZqykZ5HEtMAHJ7zBGGas4szQ/edit?tab=t.0) -- energy-efficient AI training, meeting weekly at South Park Commons (SF).

**Docs site**: https://cybertronai.github.io/SutroYaro/

## What This Is

An autonomous, multi-researcher research lab. Multiple people use different AI tools (Claude Code, Gemini CLI, Codex CLI, OpenCode, Antigravity, plain Python) to run experiments on shared challenges. A locked evaluation harness ensures comparable results. A machine-readable log accumulates findings across researchers.

Current challenge: **sparse parity** (learn XOR/parity from random {-1,+1} inputs). Next challenge: **nanoGPT** (energy-efficient training of Karpathy's nanoGPT).

## Results So Far

33 experiments, 12 wins, 15 losses, 5 inconclusive. See the [Practitioner's Field Guide](https://cybertronai.github.io/SutroYaro/research/survey/) for the full ranked results.

| Method | Time (n=20/k=3) | ARD | What it proves |
|--------|-----------------|-----|----------------|
| GF(2) Gaussian Elimination | 509 us | ~500 | Parity is linear over GF(2). 240x faster than SGD. |
| RL Bit Querying | -- | 1 | Reads exactly k=3 bits per prediction. Theoretical minimum. |
| KM Influence Estimation | 0.006s | 1,585 | O(n) not O(C(n,k)). 724x better ARD than Fourier. |
| SGD (baseline) | 0.12s | 17,976 | The neural net solves it, just the hard way. |

All 4 local learning rules (Hebbian, Predictive Coding, Equilibrium Propagation, Target Propagation) failed at chance level. Parity requires k-th order interaction detection.

## Quick Start

```bash
git clone https://github.com/cybertronai/SutroYaro.git
cd SutroYaro

# Solve 20-bit sparse parity in 0.12s (SGD)
PYTHONPATH=src python3 -m sparse_parity.fast

# Solve it in 509 microseconds (GF(2))
PYTHONPATH=src python3 src/sparse_parity/experiments/exp_gf2.py

# Run the locked evaluation harness
PYTHONPATH=src python3 src/harness.py --method gf2 --n_bits 20 --k_sparse 3

# Check your environment
PYTHONPATH=src python3 checks/env_check.py
PYTHONPATH=src python3 checks/baseline_check.py
```

## Run Autonomous Experiments

The lab runs with any AI CLI. No hooks or special setup needed -- it's a bash loop.

```bash
# Single cycle with Claude Code
bin/run-agent --tool claude --max 10

# Single cycle with Gemini CLI
bin/run-agent --tool gemini --max 10

# Overnight: 10 cycles, 5 experiments each (resilient to crashes)
bin/run-agent --loop 10 --max 5 --tool claude

# Any CLI via env var
AI_CMD="my-ai-tool -p" bin/run-agent --tool custom --max 5
```

Each cycle: fresh AI context, reads accumulated file state (log, findings, TODO), runs experiments, logs results. If a cycle crashes, the next picks up from the files.

**Antigravity** is an IDE, not a CLI -- use it manually by opening the project and following AGENT.md. For headless runs from Google's ecosystem, use Gemini CLI instead.

After a run:
```bash
bin/analyze-log          # text report with win rate, method stats
bin/analyze-log --plot   # generates results/progress.png
```

## How It Works

```
AGENT.md                  # What the AI agent follows (the loop)
src/harness.py            # Locked evaluation (agents CANNOT modify)
research/search_space.yaml  # What can be changed and to what values
research/questions.yaml   # Dependency graph of open questions
research/log.jsonl        # All 33 experiments, machine-readable
TODO.md                   # Hypothesis queue (checkboxes)
DISCOVERIES.md            # Proven facts (read before every experiment)
```

The agent reads AGENT.md, picks a hypothesis from TODO.md, designs a single-variable experiment within search_space.yaml bounds, runs it against the locked harness, classifies the result (WIN/LOSS/INVALID/INCONCLUSIVE), logs to log.jsonl, and repeats.

Safety mechanisms:
- **Harness integrity**: SHA256 verified before and after each run
- **Circuit breaker**: halts if 5+ INVALID in last 20 experiments
- **Metric isolation**: agents cannot modify tracker.py, harness.py, or data.py
- **Lock file**: PID-based, prevents concurrent cycles, detects orphaned locks

## Multi-Researcher Workflow

Multiple people run independent experiments, then merge via PR:

```
Yad (Claude Code)     Germain (Codex)      Yaroslav (Gemini CLI)
     |                      |                     |
     v                      v                     v
  log.jsonl (local)    log.jsonl (local)    log.jsonl (local)
     |                      |                     |
     +----------- PR -------+-------- PR ---------+
                            |
                    Shared log.jsonl
                    Shared DISCOVERIES.md
                    Shared scoreboard.tsv
```

```bash
# Merge a contributor's results
bin/merge-findings path/to/their-log.jsonl

# Regenerate the scoreboard
bin/merge-findings research/log.jsonl --scoreboard
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. Three levels of effort:
- **Low**: drop raw results in `contributions/` (any format)
- **Medium**: write a findings doc using `findings/_template.md`
- **High**: code + results + findings following [LAB.md](LAB.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/research/peer-research-protocol.md](docs/research/peer-research-protocol.md) for the full protocol design including the nanoGPT migration proposal.

## Project Structure

```
AGENT.md                # Agent-executable experiment loop
LAB.md                  # Human experiment protocol
DISCOVERIES.md          # Accumulated knowledge (33 proven facts)
TODO.md                 # Hypothesis queue

src/
  harness.py            # Locked evaluation harness (5 methods, CLI)
  sparse_parity/
    fast.py             # Numpy solver (0.12s)
    tracker.py          # ARD/DMC measurement (MemTracker)
    cache_tracker.py    # Cache-aware energy model
    experiments/        # All 33 experiment scripts

research/
  search_space.yaml     # Bounded mutation space per challenge
  questions.yaml        # Dependency graph of open questions
  log.jsonl             # Machine-readable experiment log

results/
  scoreboard.tsv        # Auto-generated leaderboard
  progress.png          # ARD progress chart

checks/
  env_check.py          # Pre-flight environment verification
  baseline_check.py     # Re-establish baselines per machine

bin/
  run-agent             # Tool-agnostic autonomous launcher
  merge-findings        # Import contributor results
  analyze-log           # Progress report + charts
  reproduce-all         # Verify all 14 experiments in <1s (--budget for compute caps)
  gpu_energy.py         # Real GPU energy measurement via Modal Labs (NVIDIA L4)

docs/                   # MkDocs site source
findings/               # One markdown report per experiment
```

## Links

- Docs site: https://cybertronai.github.io/SutroYaro/
- Telegram: https://t.me/sutro_group
- Main code repo: https://github.com/cybertronai/sutro
- Meetings: Mondays 18:00 at South Park Commons (380 Brannan St, SF)

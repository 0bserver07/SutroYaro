# SutroYaro

Research workspace for the [Sutro Group](https://docs.google.com/document/d/1B9867EN6Bg4ZVQK9vI_ZqykZ5HEtMAHJ7zBGGas4szQ/edit?tab=t.0) -- energy-efficient AI training, meeting weekly at South Park Commons (SF).

**Docs site**: https://cybertronai.github.io/SutroYaro/
**License**: [Unlicense](LICENSE) (Public Domain)

## Get Started

Clone the repo and open it with your coding agent. The workspace is structured so the agent can navigate it, run experiments, and report findings.

```bash
git clone https://github.com/cybertronai/SutroYaro.git
cd SutroYaro

# With Claude Code (recommended)
claude --dangerously-skip-permissions

# With Gemini CLI
gemini --yolo

# With Codex CLI
codex --full-auto
```

Then ask the agent anything: "What is this about?", "How do I run experiments?", "What are the latest findings?"

The agent reads `CLAUDE.md` (or equivalent) to understand the workspace, syncs Telegram and Google Docs for context, and can run experiments autonomously. See the [one-hour walkthrough video](https://www.youtube.com/live/L3PamTTQFGk) for a demo.

### Run experiments directly

```bash
# Solve 20-bit sparse parity in 0.12s (SGD)
PYTHONPATH=src python3 -m sparse_parity.fast

# Solve it in 509 microseconds (GF(2))
PYTHONPATH=src python3 src/sparse_parity/experiments/exp_gf2.py

# Run the locked evaluation harness
PYTHONPATH=src python3 src/harness.py --method gf2 --n_bits 20 --k_sparse 3

# Run the eval environment (Gymnasium)
PYTHONPATH=src python3 src/sparse_parity/eval/run_eval.py
```

## What This Is

A structured workspace where multiple people use different AI tools (Claude Code, Gemini CLI, Codex, Antigravity) to run experiments on shared challenges. A locked evaluation harness ensures comparable results. Findings accumulate in DISCOVERIES.md.

Current challenge: **sparse parity** (learn XOR/parity from random {-1,+1} inputs), plus sparse sum and sparse AND.

## Results So Far

36 experiments across 3 challenges. Energy metric: DMD (Data Movement Distance, Ding et al. [arXiv:2312.14441](https://arxiv.org/abs/2312.14441)), auto-measured via [TrackedArray](https://cybertronai.github.io/SutroYaro/research/tracked-numpy/).

| Method | Time | DMD | What it proves |
|--------|------|-----|----------------|
| KM-min (1 sample) | 0.050s | 3,087 | DMD leader. Parity influence is deterministic. |
| SMT Backtracking | 0.002s | 19,532 | Constraint satisfaction touches small slices. |
| KM Influence (5 samples) | 0.001s | 27,165 | O(n) not O(C(n,k)). Flip each bit, measure label change. |
| GF(2) Gaussian Elimination | 0.009s | 153,745 | Parity is linear over GF(2). |
| SGD (baseline) | 0.089s | est. | The neural net solves it, just the hard way. |

All 4 local learning rules (Hebbian, Predictive Coding, Equilibrium Propagation, Target Propagation) failed at chance level. Parity requires k-th order interaction detection.

## Eval Environment

Gymnasium-compatible environment that tests whether an AI agent can do energy-efficient ML research. The agent picks methods, observes energy metrics, gets graded on research quality.

```bash
PYTHONPATH=src python3 -c "
import gymnasium as gym; import sparse_parity.eval
env = gym.make('SutroYaro/SparseParity-v0', metric='dmc', budget=10)
obs, _ = env.reset()
obs, r, _, _, info = env.step(5)  # try GF(2)
print(f'{info[\"method\"]}: DMC={info[\"dmc\"]}, reward={r:.2f}')
"
```

- 3 challenges, 16 methods, 72-point discovery grading rubric (12 categories)
- Registry-based: add challenges and methods without editing env code
- Compute backends: local (default), Modal (GPU), remote HTTP
- Adapters: Anthropic tool-use, UK AISI Inspect
- See [AGENT_EVAL.md](AGENT_EVAL.md) for the full guide

## Recorded Sessions

| Date | Who | Title | Link |
|------|-----|-------|------|
| 2026-03-22 | Yad | Weekly catch-up, DMC experiments, parallel agents | [YouTube](https://www.youtube.com/live/L3PamTTQFGk) |
| 2026-03-16 | Yaroslav | Meeting #9: roadmap, GF(2) verification, DMC intro | [YouTube](https://www.youtube.com/watch?v=vdQ3NkEiOt8) |

Transcripts and chapters: https://cybertronai.github.io/SutroYaro/sessions/

## Weekly Catch-Up

The agent syncs Telegram, Google Docs, and GitHub into a weekly summary with action items.

Latest: https://cybertronai.github.io/SutroYaro/catchups/2026-03-22/

## Run Autonomous Experiments

```bash
# Single cycle with Claude Code
bin/run-agent --tool claude --max 10

# Single cycle with Gemini CLI
bin/run-agent --tool gemini --max 10

# Overnight: 10 cycles, 5 experiments each
bin/run-agent --loop 10 --max 5 --tool claude

# Any CLI via env var
AI_CMD="my-ai-tool -p" bin/run-agent --tool custom --max 5
```

Each cycle: fresh AI context, reads accumulated file state, runs experiments, logs results. If a cycle crashes, the next picks up from the files.

## How It Works

```
AGENT.md                  # What the AI agent follows (the loop)
AGENT_EVAL.md             # Guide for running the eval environment
LAB.md                    # Human experiment protocol
DISCOVERIES.md            # Accumulated knowledge (91 proven facts)
TODO.md                   # Hypothesis queue

src/
  harness.py              # Locked evaluation harness (5 methods, CLI)
  sparse_parity/
    fast.py               # Numpy solver (0.12s, optional tracker)
    tracker.py            # Legacy MemTracker
    lru_tracker.py        # DMD measurement (LRUStackTracker)
    tracked_numpy.py      # Auto-instrumented numpy wrapper
    cache_tracker.py      # Cache-aware energy model
    experiments/          # All 36 experiment scripts
    eval/                 # Gymnasium RL environment
      env.py              # SutroYaro/SparseParity-v0
      baselines.py        # Random, Greedy, Oracle agents
      grader.py           # 72-point discovery grading (12 categories)
      registry.py         # Extensible challenge/method registry
      backends.py         # Local, Modal, Remote compute
      answer_key.json     # 36 experiments as ground truth
      adapters/           # Anthropic, Inspect platform adapters

research/
  search_space.yaml       # Bounded mutation space per challenge
  questions.yaml          # Dependency graph of open questions
  log.jsonl               # Machine-readable experiment log

results/
  scoreboard.tsv          # Leaderboard with DMD column
  plots/                  # DMD visualizations
  eval/                   # Baseline evaluation results

docs/
  catchups/               # Weekly catch-up summaries
  sessions/               # Recorded session transcripts and chapters
  findings/               # One markdown report per experiment
  research/               # Survey, eval docs, literature
```

Safety mechanisms:
- **Harness integrity**: SHA256 verified before and after each run
- **Metric isolation**: agents cannot modify tracker.py, harness.py, or data.py
- **Circuit breaker**: halts if 5+ INVALID in last 20 experiments

## Multi-Researcher Workflow

Multiple people run independent experiments, then merge via PR:

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

## Links

- Docs site: https://cybertronai.github.io/SutroYaro/
- Eval environment: [docs](https://cybertronai.github.io/SutroYaro/research/eval-environment/)
- Sessions: [transcripts and chapters](https://cybertronai.github.io/SutroYaro/sessions/)
- Weekly catch-ups: [latest](https://cybertronai.github.io/SutroYaro/catchups/2026-03-22/)
- Telegram: https://t.me/sutro_group
- Main code repo: https://github.com/cybertronai/sutro
- Meetings: Mondays 18:00 at South Park Commons (380 Brannan St, SF)

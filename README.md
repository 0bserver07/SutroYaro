# SutroYaro

Research workspace for the [Sutro Group](https://t.me/sutro_group) — energy-efficient AI training, meeting weekly at South Park Commons (SF).

**Docs site**: https://0bserver07.github.io/SutroYaro/

## What This Is

An autonomous research lab for the sparse parity challenge: learn XOR/parity from random numbers, scale to 20+ bits with noise, and measure/improve energy efficiency via Average Reuse Distance (ARD).

We use Claude Code with parallel agent teams to run experiments, record findings, and accumulate knowledge automatically.

## Results

**20-bit sparse parity (k=3) solved** — 100% accuracy in 0.12 seconds (numpy) across 5 seeds.

- LR=0.1, batch=32, hidden=200, n_train=1000
- Classic grokking pattern: hidden progress invisible to loss metrics, then sharp phase transition
- Per-layer forward-backward gives 3.8% ARD improvement for free
- Forward-Forward algorithm has 25x worse ARD than backprop (opposite of hypothesis)
- Curriculum learning (n=10→30→50) gives 14.6x speedup, cracks n=50 which direct training can't
- Sign SGD solves k=5 2x faster; standard SGD also works with enough data (n_train=5000)
- Weight decay 0.01 optimal, narrow working range [0.01, 0.05]

See [DISCOVERIES.md](DISCOVERIES.md) for the full knowledge base and [changelog](https://0bserver07.github.io/SutroYaro/changelog/) for version history.

## Quick Start

```bash
# Run the fast solver (numpy, <0.2s)
PYTHONPATH=src python3 -m sparse_parity.fast

# Run the full pipeline (pure Python, 3 training variants, ARD comparison)
PYTHONPATH=src python3 -m sparse_parity.run

# Run tests
python3 -m pytest tests/ -v

# Run a specific experiment
PYTHONPATH=src python3 src/sparse_parity/experiments/exp_sign_sgd.py
```

## Running Your Own Experiments

This repo is set up for autonomous AI-assisted research. See [LAB.md](LAB.md) for the full protocol.

1. Read [DISCOVERIES.md](DISCOVERIES.md) — what's proven, what's open
2. Pick an open question
3. Copy `src/sparse_parity/experiments/_template.py`
4. Run, measure, write findings, commit
5. Update DISCOVERIES.md with your result

Each experiment produces:
- `src/sparse_parity/experiments/exp_{name}.py` — the code
- `findings/exp_{name}.md` — analysis (strict template)
- `results/exp_{name}/results.json` — machine-readable metrics

## How Agent-Driven Research Works

```mermaid
flowchart TD
    H["Human (You)"] -->|writes specs| Docs
    H -->|prompts| Lead

    subgraph Repo["GitHub Repo (SutroYaro)"]
        Docs["CLAUDE.md · DISCOVERIES.md · LAB.md\nproposed-approaches.md"]
        Scripts["sync_telegram.ts · sync_google_docs.py\nexport_sessions.py"]
    end

    Lead["Lead Agent\n(Claude Code)"] -->|reads first| Docs
    Scripts -->|feeds context| Lead

    Lead -->|1. reads docs| Step1["Survey problem space\n(27 approaches)"]
    Step1 -->|2. dispatches| Agents

    subgraph Agents["Isolated Sub-Agents (x17, parallel)"]
        A1["GF(2)"]
        A2["Hebbian"]
        A3["RL Bandit"]
        A4["..."]
    end

    Agents -->|each produces| Out["experiment.py · results.json · findings.md"]
    Out -->|feeds back into| Docs

    style H fill:#fff2cc,stroke:#d6b656,color:#333
    style Repo fill:#f0f4ff,stroke:#6c8ebf,color:#333
    style Docs fill:#e1d5e7,stroke:#9673a6,color:#333
    style Scripts fill:#d5e8d4,stroke:#82b366,color:#333
    style Lead fill:#ffe6cc,stroke:#d79b00,color:#333
    style Step1 fill:#fff2cc,stroke:#d6b656,color:#333
    style Agents fill:#fce4ec,stroke:#b85450,color:#333
    style A1 fill:#f8cecc,stroke:#b85450,color:#333
    style A2 fill:#f8cecc,stroke:#b85450,color:#333
    style A3 fill:#f8cecc,stroke:#b85450,color:#333
    style A4 fill:#f8cecc,stroke:#b85450,color:#333
    style Out fill:#d5e8d4,stroke:#82b366,color:#333
```

The human writes specs (CLAUDE.md, DISCOVERIES.md, LAB.md). The lead agent reads those first, surveys the problem space, then dispatches isolated sub-agents in parallel. Each sub-agent gets one approach, the experiment template, and shared modules. No sub-agent sees another's results or the knowledge base. Read-only access to benchmark code prevents agents from gaming the metrics. Outputs feed back into DISCOVERIES.md for the next round.

Three automation scripts feed live context to the agents:

- `sync_google_docs.py` pulls the group's Google Docs into local markdown
- `sync_telegram.ts` pulls Telegram thread messages into JSON (one-way, read-only)
- `.traces/export_sessions.py` exports agent conversation traces with timestamps (audit trail)

The files that make this work:

- `CLAUDE.md` — problem context, constraints, current best config
- `DISCOVERIES.md` — accumulated proven facts from all experiments (shared memory across agents)
- `LAB.md` — experiment protocol, templates, lifecycle
- `proposed-approaches.md` — candidate methods generated during survey step
- `_template.py` — experiment code starter
- `findings/*.md` — structured experiment reports

For parallel execution, we use Claude Code's team agents feature: a lead agent reads DISCOVERIES.md, creates tasks, and dispatches worker agents that each run an independent experiment.

See [findings/prompting-strategies.md](findings/prompting-strategies.md) for detailed prompting lessons.

## Structure

```
LAB.md              # Lab protocol for autonomous sessions
DISCOVERIES.md      # Accumulated knowledge base
TODO.md             # Open research tasks

src/sparse_parity/
  fast.py           # Numpy-accelerated solver (0.12s)
  config.py         # Experiment configuration
  data.py           # Dataset generation
  model.py          # MLP model + forward pass
  tracker.py        # ARD measurement (MemTracker)
  cache_tracker.py  # Cache-aware ARD (CacheTracker)
  train.py          # Standard backprop
  train_fused.py    # Fused layer-wise updates
  train_perlayer.py # Per-layer forward-backward
  run.py            # Full pipeline runner
  experiments/      # All experiment scripts

findings/           # Experiment reports (13 so far)
results/            # JSON metrics per experiment
research/           # Literature review
docs/               # MkDocs site source
```

## Links

- Docs site: https://0bserver07.github.io/SutroYaro/
- Telegram: https://t.me/sutro_group
- Main code repo: https://github.com/cybertronai/sutro
- Meetings: Mondays 18:00 at South Park Commons (380 Brannan St, SF)

# CLAUDE.md - Sutro Group Research Workspace

## Project Context

This is a research workspace for the **Sutro Group**, a study group exploring energy-efficient AI training. The group meets weekly at South Park Commons in San Francisco.

## Read These First

- **LAB.md** — Protocol for running experiments (templates, lifecycle, rules)
- **DISCOVERIES.md** — What's proven so far (read before every experiment)
- **TODO.md** — Open research tasks

## Core Concepts

- **Sparse Parity**: The benchmark task — learn XOR/parity from random positive/negative numbers. n=20 bits, k=3 relevant, 17 noise. Solved at 100% in 0.12s.
- **Average Reuse Distance (ARD)**: Proxy metric for energy efficiency. Small ARD = data stays in cache = cheap. Large ARD = expensive external memory access.
- **Grokking**: Phase transition in training — loss looks flat for many epochs, then accuracy suddenly jumps. Hidden progress visible in ||w_t - w_0||_1.
- **CacheTracker**: Extended MemTracker with LRU cache simulation for realistic energy estimates.

## Current Best Config (20-bit, k=3)

```python
n_bits=20, k_sparse=3, hidden=200, lr=0.1, wd=0.01,
batch_size=32, n_train=1000, max_epochs=200
```

Solves in ~40 epochs / 0.12s with numpy (`fast.py`).

## Key Findings

- LR=0.1 is critical (0.5 overshoots, never triggers phase transition)
- Per-layer forward-backward gives 3.8% ARD improvement for free
- Forward-Forward has 25x WORSE ARD than backprop for 2-layer networks
- Curriculum learning (n=10→30→50) gives 14.6x speedup at scale
- Sign SGD solves k=5, standard SGD also works with enough data (n_train=5000)
- ARD metric doesn't model cache — batch looks worse but is better on real hardware
- WD=0.01 optimal, narrow range [0.01, 0.05]
- SGD breaks at n^k > 100,000 iterations

## Working Style

- Iteration time must stay under 2 seconds (use `fast.py` for numpy speed)
- Change one thing at a time (correctness, then speed, then energy)
- Priority: correctness > wall-clock time > energy usage
- One hypothesis per experiment, always compare against baseline
- Record everything — failed hypotheses are findings too

## Key People

- **Yad** (repo creator, SutroYaro) — Built the Claude Code autonomous research lab, parallel agent experiments
- **Yaroslav** (Sutro Group founder) — Technical sprints, algorithm work, cybertronai/sutro
- **Emmett** — Aster agentic loop framework, 2x energy improvement on microgpt
- **Germaine**, **Andy**, **Seth**, **Barak**, **Jamie Simon** — Group members

## Related Repos

- https://github.com/cybertronai/sutro — Main code repo with sparse_parity_benchmark.py
- https://github.com/0bserver07/SutroYaro — This research workspace

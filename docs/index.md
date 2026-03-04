# SutroYaro

Research workspace for the **Sutro Group** -- energy-efficient AI training, meeting weekly at South Park Commons (SF).

## Status: Sparse Parity Challenge SOLVED

20-bit sparse parity (k=3, 17 noise bits) solved at **100% accuracy in 0.12 seconds** across 5 random seeds. 16 experiments completed across 3 rounds of research.

!!! success "Key Results"
    - **LR=0.1, batch=32, hidden=200** — correct hyperparams were the main fix (not the algorithm)
    - **Per-layer forward-backward** gives 3.8% ARD improvement for free
    - **Curriculum learning** (n=10→30→50) gives 14.6x speedup, cracks n=50
    - **Forward-Forward** has 25x worse ARD than backprop — not viable for small networks
    - **For small k, sparse parity is a search problem** — Fourier/random search is 13x faster than SGD

!!! info "Energy Insight"
    Memory cost is the biggest contributor to energy use in training. Local registers cost ~5pJ vs HBM at ~640pJ — a 128x difference.

## Quick Start

```bash
# Solve 20-bit sparse parity in 0.12s
PYTHONPATH=src python3 -m sparse_parity.fast

# Run the full pipeline (3 training variants, ARD comparison)
PYTHONPATH=src python3 -m sparse_parity.run

# Run your own experiment
cp src/sparse_parity/experiments/_template.py src/sparse_parity/experiments/exp_mine.py
# Edit, run, record findings
```

## Navigation

- [Learning Guide](learning-guide.md) — concepts explained from scratch (start here if new)
- [Discoveries](https://github.com/0bserver07/SutroYaro/blob/main/DISCOVERIES.md) — accumulated knowledge from all experiments
- [Changelog](changelog.md) — version history with all results
- [Lab Protocol](https://github.com/0bserver07/SutroYaro/blob/main/LAB.md) — how to run autonomous experiments

## Quick Links

| Resource | Link |
|----------|------|
| Telegram | [t.me/sutro_group](https://t.me/sutro_group) |
| Code repo | [cybertronai/sutro](https://github.com/cybertronai/sutro) |
| Meetings | Mondays 18:00 at South Park Commons (380 Brannan St) |
| Bill Daly talk | [Energy use in GPUs](https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457) |

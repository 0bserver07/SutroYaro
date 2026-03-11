# SutroYaro

Research workspace for the **Sutro Group**, a study group exploring energy-efficient AI training. Weekly meetings at South Park Commons, San Francisco.

## The Challenge

The group picks simple learning tasks and tries to solve them with less energy. Challenge #1 is **sparse parity**: given n-bit inputs in {-1, +1}, find the k secret bits whose product determines the label. It is the simplest non-trivial learning problem, fast enough to iterate hundreds of times per hour, and structured enough to reveal real phenomena about memory access patterns and algorithm design.

Standard config: n=20 bits, k=3 secret, 17 noise. Harder configs: n=50/k=3, n=20/k=5, n=100/k=10.

## What We Found

33 experiments across two phases. The full ranked results and methodology are in the [Practitioner's Field Guide](research/survey.md).

**Phase 1** (16 experiments): Started with a broken SGD baseline (LR=0.5, stuck at 54%). Fixed hyperparameters to solve it in 0.12s. Optimized memory access patterns (ARD) within the SGD framework, hitting a ceiling at ~10% improvement because one tensor (W1) dominates 75% of all float reads. Built a cache simulator showing L2 eliminates all misses. Pivoted to new algorithms.

**Phase 2** (17 experiments): Tested algebraic, information-theoretic, local learning, hardware-aware, and alternative approaches in parallel. The result:

| Method | Time (n=20/k=3) | Why it works |
|--------|-----------------|-------------|
| GF(2) Gaussian Elimination | 509 us | Parity is linear over the binary field. Row-reduce. |
| Kushilevitz-Mansour | 0.001-0.006s | Flip each bit, measure label change. Secret bits have influence 1.0. |
| SMT Backtracking | 0.002s | Constraint satisfaction with k-1 pruning. |
| SGD (baseline) | 0.12s | The neural network solves it, just 240x slower. |

All four local learning rules (Hebbian, Predictive Coding, Equilibrium Propagation, Target Propagation) failed at chance level. Parity requires k-th order interaction detection, which local statistics cannot provide.

## Quick Start

```bash
# Clone and run
git clone https://github.com/0bserver07/SutroYaro.git
cd SutroYaro

# Solve 20-bit sparse parity in 0.12s (SGD)
PYTHONPATH=src python3 -m sparse_parity.fast

# Solve it in 509 microseconds (GF(2))
PYTHONPATH=src python3 src/sparse_parity/experiments/exp_gf2.py

# Run your own experiment
cp src/sparse_parity/experiments/_template.py src/sparse_parity/experiments/exp_mine.py
```

## Where to Find Things

| What | Where |
|------|-------|
| Full results and methodology | [Practitioner's Field Guide](research/survey.md) |
| Autonomous research protocol | [Peer Research Protocol](research/peer-research-protocol.md) |
| What's been proven so far | [DISCOVERIES.md](https://github.com/0bserver07/SutroYaro/blob/main/DISCOVERIES.md) |
| Individual experiment findings | [Research > Findings](research/index.md) |
| Group context and people | [Context](context.md) |
| Goals and success criteria | [Goals](goals.md) |
| How we use Claude Code | [Tooling](tooling/index.md) |
| Meeting notes and Google Docs | [Meetings](meetings/index.md) |
| How this project uses AI agents | [AGENTS.md](https://github.com/0bserver07/SutroYaro/blob/main/AGENTS.md) |
| How to contribute | [CONTRIBUTING.md](https://github.com/0bserver07/SutroYaro/blob/main/CONTRIBUTING.md) |

## Quick Links

| Resource | Link |
|----------|------|
| Telegram | [t.me/sutro_group](https://t.me/sutro_group) |
| Code repo | [cybertronai/sutro](https://github.com/cybertronai/sutro) |
| Meetings | Mondays 18:00 at South Park Commons (380 Brannan St) |

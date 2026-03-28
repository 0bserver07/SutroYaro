# SutroYaro

Research workspace for the **Sutro Group**, a study group exploring energy-efficient AI training. Weekly meetings at South Park Commons, San Francisco.

## The Challenges

The group picks simple learning tasks and tries to solve them with less energy. Three challenges so far, all runnable in under 1 second:

| Challenge | Task | What it tests |
|-----------|------|---------------|
| **Sparse Parity** | y = product of k secret bits (XOR) | k-th order interactions, no local signal |
| **Sparse Sum** | y = sum of k secret bits | First-order linear structure |
| **Sparse AND** | y = logical AND of k secret bits | Class-imbalanced k-th order |

Standard config: n=20 bits, k=3 secret, 17 noise. The [adding-a-challenge guide](research/adding-a-challenge.md) documents how any agent or contributor can add the next task.

## What We Found

36 experiments across three phases, plus GPU energy validation. The full ranked results are in the [Practitioner's Field Guide](research/survey.md).

**Phase 1** (16 experiments): Started with a broken SGD baseline (LR=0.5, stuck at 54%). Fixed hyperparameters to solve it in 0.12s. Optimized memory access patterns (ARD) within the SGD framework, hitting a ceiling at ~10% improvement because one tensor (W1) dominates 75% of all float reads. Pivoted to new algorithms.

**Phase 2** (17 experiments): Tested algebraic, information-theoretic, local learning, hardware-aware, and alternative approaches in parallel:

| Method | Time (n=20/k=3) | Why it works |
|--------|-----------------|-------------|
| GF(2) Gaussian Elimination | 509 us | Parity is linear over the binary field. Row-reduce. |
| Kushilevitz-Mansour | 0.001-0.006s | Flip each bit, measure label change. Secret bits have influence 1.0. |
| SMT Backtracking | 0.002s | Constraint satisfaction with k-1 pruning. |
| SGD (baseline) | 0.12s | The neural network solves it, just 240x slower. |

All four local learning rules (Hebbian, Predictive Coding, Equilibrium Propagation, Target Propagation) failed at chance level. Parity requires k-th order interaction detection, which local statistics cannot provide.

**GPU measurement**: Ran methods on NVIDIA L4 via Modal Labs using PyTorch CUDA (5 runs). GPU is 4-790x slower than CPU at this problem size. Sparse parity tensors are too small for CUDA to help. See [GPU vs CPU findings](findings/exp_proxy_comparison.md).

## Quick Start

```bash
git clone https://github.com/cybertronai/SutroYaro.git
cd SutroYaro

# Verify all 14 experiments across 3 challenges in <1 second
PYTHONPATH=src python3 bin/reproduce-all

# Run sparse parity with GF(2) (509 microseconds)
PYTHONPATH=src python3 src/harness.py --method gf2 --n_bits 20 --k_sparse 3

# Run sparse sum (new challenge)
PYTHONPATH=src python3 src/harness.py --challenge sparse-sum --method sgd

# Measure real GPU energy via Modal Labs
pip install modal && modal token set
modal run bin/gpu_energy.py

# Run autonomous agent loop
bin/run-agent --tool claude --max 10
```

## Where to Find Things

| What | Where |
|------|-------|
| **New here? Start here** | [What's New (March 2026)](research/whats-new-march-2026.md) |
| All 36 experiments ranked | [Practitioner's Field Guide](research/survey.md) |
| Add a new challenge | [Adding a Challenge](research/adding-a-challenge.md) |
| GPU vs CPU findings | [GPU vs CPU for Sparse Parity](findings/exp_proxy_comparison.md) |
| Run experiments with any AI tool | [Agent CLI Guide](tooling/agent-cli-guide.md) |
| Scripts and toolkit | [Tooling](tooling/index.md) |
| Full protocol design | [Peer Research Protocol](research/peer-research-protocol.md) |
| What's been proven so far | [DISCOVERIES.md](https://github.com/cybertronai/SutroYaro/blob/main/DISCOVERIES.md) |
| Individual experiment findings | [Research > Findings](research/index.md) |
| Meeting notes and Google Docs | [Meetings](meetings/index.md) |
| Auto DMD tracking | [Auto-instrumented DMD Tracking](research/tracked-numpy.md) |
| How to contribute | [CONTRIBUTING.md](https://github.com/cybertronai/SutroYaro/blob/main/CONTRIBUTING.md) |

## Quick Links

| Resource | Link |
|----------|------|
| Telegram | [t.me/sutro_group](https://t.me/sutro_group) |
| Code repo | [cybertronai/sutro](https://github.com/cybertronai/sutro) |
| The Bigger Picture | [Yaroslav's roadmap](google-docs/bigger-picture.md) |
| Meetings | Mondays 18:00 at South Park Commons (380 Brannan St) |

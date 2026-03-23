# Experiment: GrokFast v2 — Hard Regime Testing

**Date**: 2026-03-23
**Status**: COMPLETED — WIN on k=5, LOSS on n=30/k=3, NEUTRAL on n=20/k=3
**Researcher**: Seth Stafford

## Question

Does GrokFast accelerate grokking on harder sparse parity configurations where the
phase transition is genuinely delayed? (exp4 showed it was counterproductive on the
easy n=20/k=3 regime.)

## Hypothesis

GrokFast amplifies slow gradient components via EMA filtering. It should help when
the grokking plateau is long (high-k problems with k-th order interactions), even
if it hurts on easy problems where SGD already converges fast.

## What was performed

Tested SGD vs GrokFast with 3 hyperparameter settings across 3 difficulty regimes,
5 seeds each (75 total runs). All use numpy-accelerated mini-batch SGD (batch_size=32).

| Parameter | easy_n20_k3 | hard_n30_k3 | hard_n20_k5 |
|-----------|-------------|-------------|-------------|
| n_bits | 20 | 30 | 20 |
| k_sparse | 3 | 3 | 5 |
| hidden | 200 | 200 | 200 |
| lr | 0.1 | 0.1 | 0.1 |
| wd | 0.01 | 0.01 | 0.01 |
| max_epochs | 200 | 200 | 500 |
| n_train | 1000 | 1000 | 2000 |
| batch_size | 32 | 32 | 32 |

GrokFast settings tested:
- **(a=0.98, l=2.0)**: Original paper defaults — aggressive
- **(a=0.95, l=1.0)**: Less aggressive smoothing and amplification
- **(a=0.99, l=0.5)**: High smoothing, gentle amplification

## What was produced

### n=20, k=3 (easy — confirms exp4)

| Method | Solve Rate | Avg Epoch | Avg Time |
|--------|-----------|-----------|----------|
| SGD | 100% | 39 | 0.08s |
| GrokFast(0.98, 2.0) | **80%** | 200 | 0.17s |
| GrokFast(0.95, 1.0) | 100% | 34 | 0.07s |
| GrokFast(0.99, 0.5) | 100% | 37 | 0.12s |

Aggressive GrokFast hurts. Mild settings are neutral.

### n=30, k=3 (more noise dimensions)

| Method | Solve Rate | Avg Epoch | Avg Time |
|--------|-----------|-----------|----------|
| SGD | 100% | 91 | 0.19s |
| GrokFast(0.98, 2.0) | **40%** | 200 | 0.25s |
| GrokFast(0.95, 1.0) | 100% | 85 | 0.20s |
| GrokFast(0.99, 0.5) | 100% | 95 | 0.24s |

Aggressive GrokFast is even worse with more noise dimensions (40% solve rate).
Mild settings are again neutral — slightly fewer epochs but similar or worse wall time.

### n=20, k=5 (higher-order interactions)

| Method | Solve Rate | Avg Epoch | Avg Time |
|--------|-----------|-----------|----------|
| SGD | 100% | 73 | 0.35s |
| **GrokFast(0.98, 2.0)** | **100%** | **29** | **0.15s** |
| GrokFast(0.95, 1.0) | 100% | 37 | 0.19s |
| GrokFast(0.99, 0.5) | 100% | 54 | 0.29s |

**GrokFast wins decisively.** The aggressive setting (a=0.98, l=2.0) gives 2.5x
fewer epochs and 2.3x faster wall time. All three GrokFast settings outperform SGD.

## Can it be reproduced?

Yes. 5 seeds per configuration, all 100% solve rate on the winning regime.
Script: `src/sparse_parity/experiments/exp_grokfast_v2.py`
Results: `results/exp_grokfast_v2/results.json`

## Finding

**GrokFast accelerates grokking when k is large (higher-order interactions create a
genuinely long plateau) but is harmful or neutral when n is large (more noise
dimensions to amplify).** The critical variable is interaction order, not input
dimension. On n=20/k=5, aggressive GrokFast (a=0.98, l=2.0) gives a 2.5x epoch
reduction and 2.3x wall-time speedup over vanilla SGD.

## Analysis

### Why k matters more than n

GrokFast amplifies slowly-evolving gradient components. For sparse parity:
- **High k**: The network must discover a k-th order interaction. The gradient signal
  for the correct feature combination is exponentially weak early in training
  (proportional to 1/2^k). GrokFast accumulates this weak signal over time via the
  EMA, effectively boosting signal-to-noise.
- **High n with low k**: More noise dimensions means the EMA accumulates noise too.
  With k=3, the gradient signal is already strong enough that amplification adds
  more noise than signal.

### The aggressive setting is polarized

(a=0.98, l=2.0) is either the best or worst setting depending on the regime. This
makes sense: strong amplification helps when the signal is genuinely weak (k=5) but
causes instability when the signal is already adequate (k=3).

### Mild GrokFast is never worse than SGD

(a=0.95, l=1.0) matches or slightly beats SGD across all regimes. This could be a
safe default for unknown problem difficulty.

## Open questions

- Does GrokFast + curriculum compound? Curriculum shortens the plateau from the
  input-dimension side; GrokFast shortens it from the interaction-order side.
- What happens at k=7 or k=10? The speedup may grow with k.
- Can we adaptively tune lambda based on gradient variance during training?

## Files

- Experiment script: `src/sparse_parity/experiments/exp_grokfast_v2.py`
- Results JSON: `results/exp_grokfast_v2/results.json`

## References

- Lee et al. 2024, "GrokFast: Accelerated Grokking by Amplifying Slow Gradients" — https://arxiv.org/abs/2405.20233
- exp4 (previous GrokFast test on easy regime): `findings/exp4_grokfast.md`

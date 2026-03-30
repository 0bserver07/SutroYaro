# Experiment: GrokFast + Curriculum DMD Measurement

**Date**: 2026-03-30
**Status**: COMPLETED — GF+Curriculum gives 83x DMD reduction on n=50/k=5
**Researcher**: SethTS

## Question

What is the actual data movement cost (DMD) of our GrokFast + Curriculum methods?
Does GrokFast's epoch reduction offset its EMA buffer overhead in total DMD?

## Hypothesis

GrokFast adds 4 EMA buffers (ema_W1, ema_b1, ema_W2, ema_b2) that increase per-step
DMD. But if it converges in fewer epochs, total DMD should still be lower. Curriculum
training at small n should have much lower per-step DMD because array sizes are smaller.

## What was performed

Measured per-step DMD for 4 methods across 3 regimes using buffer-level LRU stack
tracking. Per-step DMD × total steps from convergence experiments (SethTS-001/002)
gives total DMD.

Note: per-element TrackedArray tracking was attempted first but is O(n^2) with the
list-based LRU stack (see splay tree optimization below). Buffer-level tracking
gives true LRU ordering at buffer granularity.

## What was produced

### Per-step DMD: GrokFast costs ~56% more per step

| Method | DMD/step (n=20) | DMD/step (n=50) | Overhead |
|--------|----------------|----------------|----------|
| SGD | 91,606 | 142,612 | baseline |
| GrokFast | 143,078 | 263,891 | ~56-85% |

The EMA buffers add 4 extra read-write pairs per step, pushing data deeper in the
LRU stack and increasing stack distances for subsequent reads.

### Total DMD: curriculum dominates

| Regime | Method | Epochs | Total DMD | vs SGD |
|--------|--------|--------|-----------|--------|
| n=20/k=5 | SGD | 70 | 397M | 1.00x |
| n=20/k=5 | GrokFast | 25 | 222M | 0.56x |
| n=20/k=5 | Curriculum | 25 | 115M | 0.29x |
| n=20/k=5 | **GF+Curriculum** | **12** | **80M** | **0.20x** |
| n=50/k=3 | SGD | 58 | 513M | 1.00x |
| n=50/k=3 | GrokFast | 148 | 2,421M | 4.72x |
| n=50/k=3 | Curriculum | 10 | 55M | 0.11x |
| n=50/k=3 | GF+Curriculum | 7 | 65M | 0.13x |
| n=50/k=5 | SGD | 1000 | 8,949M | 1.00x (FAIL) |
| n=50/k=5 | GrokFast | 1000 | 16,468M | 1.84x (FAIL) |
| n=50/k=5 | Curriculum | 34 | 163M | 0.02x |
| n=50/k=5 | **GF+Curriculum** | **14** | **108M** | **0.01x** |

## Can it be reproduced?

Yes. Script: `src/sparse_parity/experiments/exp_grokfast_curriculum_dmd.py`
Results: `results/exp_grokfast_curriculum_dmd/results.json`

## Finding

**GrokFast has ~56% higher per-step DMD than SGD** due to EMA buffer overhead, but
total DMD is still lower when it converges faster. **Curriculum is the dominant factor**
in DMD reduction because training at small n (n=10) has per-step DMD 35-45% lower
than at the target n. GF+Curriculum on n=50/k=5 achieves an **83x total DMD reduction**
while going from 0% to 100% accuracy.

**GrokFast alone is a DMD disaster on large n**: 4.72x worse than SGD on n=50/k=3
(more epochs AND higher per-step cost). The EMA overhead compounds when convergence
is slow.

## Analysis

### Why curriculum dominates DMD

At n=10, W1 is 200×10=2000 floats vs 200×50=10000 at n=50. Since DMD =
`size × sqrt(stack_distance)`, smaller arrays contribute less per-step. And curriculum
needs only 1-2 epochs at the large n (expansion is nearly free), so almost all DMD
cost is paid at the cheap small-n phase.

### The GrokFast tradeoff

GrokFast is worth the per-step overhead when k is large (convergence speedup outweighs
EMA cost) but harmful when k is small and n is large (no convergence benefit, pure
overhead). This matches the epoch-level findings from SethTS-001.

## Open questions

- Per-element tracking would give more precise DMD (accounts for partial buffer reads
  during masking). The splay tree optimization in this PR makes this feasible.
- How does DMD correlate with actual energy measurements via pynvml or CodeCarbon?

## Files

- Experiment script: `src/sparse_parity/experiments/exp_grokfast_curriculum_dmd.py`
- Results JSON: `results/exp_grokfast_curriculum_dmd/results.json`

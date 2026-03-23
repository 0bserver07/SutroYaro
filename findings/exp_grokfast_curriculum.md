# Experiment: GrokFast + Curriculum — Compounding Test

**Date**: 2026-03-23
**Status**: COMPLETED — WIN, gains compound on all 3 regimes
**Researcher**: SethTS

## Question

Do GrokFast and curriculum learning compound? They attack different axes:
curriculum neutralizes n-scaling (noise dimensions), GrokFast accelerates
k-th order grokking (interaction order). If they're orthogonal, combining
them should multiply the speedups.

## Hypothesis

GrokFast + n-curriculum will outperform either method alone, especially
on n=50/k=5 where both n and k are hard.

## What was performed

4 methods across 3 difficulty regimes, 5 seeds each (60 total runs).
GrokFast uses aggressive setting (a=0.98, l=2.0) which won on k=5 in exp_grokfast_v2.

| Parameter | All regimes |
|-----------|-------------|
| hidden | 200 |
| lr | 0.1 |
| wd | 0.01 |
| batch_size | 32 |
| n_train | 2000 |
| n_test | 200 |
| GrokFast alpha | 0.98 |
| GrokFast lambda | 2.0 |
| seeds | 42-46 |

Curriculum stages: [10, 20] for n=20 target, [10, 30, 50] for n=50 target.
All secrets have indices < 10 so they're valid at the smallest curriculum stage.

## What was produced

### n=20, k=5 (GrokFast's strength)

| Method | Solve Rate | Avg Epochs | Avg Time | vs SGD |
|--------|-----------|------------|----------|--------|
| SGD | 100% | 70 | 0.284s | baseline |
| GrokFast | 100% | 25 | 0.134s | 2.8x |
| Curriculum [10->20] | 100% | 25 | 0.094s | 2.8x |
| **GrokFast + Curriculum** | **100%** | **12** | **0.057s** | **5.8x** |

### n=50, k=3 (Curriculum's strength)

| Method | Solve Rate | Avg Epochs | Avg Time | vs SGD |
|--------|-----------|------------|----------|--------|
| SGD | 100% | 58 | 0.298s | baseline |
| GrokFast | 80% | 148 | 0.434s | worse |
| Curriculum [10->30->50] | 100% | 10 | 0.055s | 5.8x |
| **GrokFast + Curriculum** | **100%** | **7** | **0.035s** | **8.3x** |

### n=50, k=5 (both hard -- the real test)

| Method | Solve Rate | Avg Epochs | Avg Time | vs SGD |
|--------|-----------|------------|----------|--------|
| SGD | **0%** | 1000 | 5.426s | FAIL |
| GrokFast | **0%** | 1000 | 6.121s | FAIL |
| Curriculum [10->30->50] | 100% | 34 | 0.153s | solves |
| **GrokFast + Curriculum** | **100%** | **14** | **0.077s** | **solves 2.4x faster** |

## Can it be reproduced?

Yes. 5 seeds per method per regime, 100% solve rate on all combined runs.
Script: `src/sparse_parity/experiments/exp_grokfast_curriculum.py`
Results: `results/exp_grokfast_curriculum/results.json`

## Finding

**GrokFast and curriculum compound on all three regimes.** The combination
always outperforms either method alone. On n=50/k=5, which is completely
unsolvable by SGD (0% at 1000 epochs), GrokFast + Curriculum solves it in
14 epochs / 77ms. The two methods are orthogonal: curriculum handles the
n-scaling wall by transferring learned features, GrokFast accelerates the
k-th order grokking plateau by amplifying weak gradient signals.

## Analysis

### Why they compound

Curriculum eliminates most of the n-related difficulty: the network learns
"which bits matter" at small n where it's cheap, then expanding to large n
is nearly free (1-2 epochs per expansion). But curriculum doesn't help with
the k-th order interaction discovery at the initial small-n phase. That's
where GrokFast helps: even at n=10, finding a k=5 interaction requires
detecting a weak 5th-order gradient signal, and GrokFast's EMA accumulation
accelerates this.

The multiplication is nearly clean:
- n=20/k=5: GrokFast alone = 2.8x, Curriculum alone = 2.8x, Combined = 5.8x (vs 7.8x ideal)
- n=50/k=5: Curriculum alone = 29x vs SGD (solving vs failing), GrokFast adds another 2.4x on top

### GrokFast still hurts without curriculum on large n

On n=50/k=3, GrokFast alone is worse than SGD (80% solve rate, more epochs).
But GrokFast + Curriculum beats curriculum alone (7 vs 10 epochs). The
curriculum shields GrokFast from the noise-dimension problem by keeping n
small during the critical learning phase.

### The n=50/k=5 result is notable

This configuration was previously considered impractical for SGD (see DISCOVERIES.md:
"standard SGD breaks at n^k > 100,000 steps"). Neither SGD nor GrokFast alone can
solve it. Curriculum alone takes 34 epochs. The combination takes 14 epochs at 77ms
wall time, 100% solve rate.

## Open questions

- Does this scale further? n=100/k=5 or n=50/k=7?
- What's the DMC profile? Curriculum uses fewer total epochs but each expansion
  phase reads W1 at a larger size. GrokFast adds EMA buffers (4 extra arrays).
- Can we adapt lambda during curriculum phases (aggressive at small n, mild at large n)?

## Files

- Experiment script: `src/sparse_parity/experiments/exp_grokfast_curriculum.py`
- Results JSON: `results/exp_grokfast_curriculum/results.json`

## References

- exp_grokfast_v2: GrokFast regime testing (this PR)
- exp_curriculum: Original curriculum experiment
- Lee et al. 2024, "GrokFast: Accelerated Grokking" — https://arxiv.org/abs/2405.20233

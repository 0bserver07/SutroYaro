# Experiment: DMC Optimization for Sparse Parity

**Date**: 2026-03-22
**Issue:** #22
**Status**: SUCCESS

## Hypothesis

If we reduce the number of influence samples in KM from 5 to 1 (exploiting that parity influence is exactly 0 or 1, never fractional) and use in-place buffer reuse to minimize stack distances, we can reduce DMC below the GF2 baseline of 8,607.

**Result:** SUCCESS -- best variant achieves DMC 3,578 (58% reduction from GF2 baseline).

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20 |
| k_sparse | 3 |
| seed | 42 (+ robustness: 43, 44, 45, 46) |
| method | KM influence estimation variants |

## Approaches Tested

Five DMC optimization strategies, compared against GF2 (DMC=8,607) and KM (DMC=20,633) baselines:

- **A. KM-min**: 1 influence sample per bit instead of 5. Separate x and x_flip buffers.
- **B. KM-shared**: 5 samples but shared buffer names (same name reused across bits).
- **C. GF2 fine-grained**: Track each Gaussian elimination row operation individually.
- **D. KM-min + verify**: 1 influence sample + minimal (k+1=4) verification samples.
- **E. KM-inplace**: 1 influence sample, single buffer written once and read twice per bit (no separate flip buffer).

## Results

| Method | Accuracy | ARD | DMC | Total Floats | vs GF2 |
|--------|----------|-----|-----|-------------|--------|
| GF2 baseline | 1.00 | 420 | 8,607 | 860 | 1.00x |
| KM baseline | 1.00 | 92 | 20,633 | 4,420 | 2.40x |
| **KM-min (A)** | **1.00** | **20** | **3,578** | **1,600** | **0.42x** |
| KM-shared (B) | 1.00 | 92 | 20,633 | 4,400 | 2.40x |
| GF2 fine-grained (C) | 1.00 | 463 | 189,056 | 17,660 | 21.96x |
| KM-min + verify (D) | 1.00 | 26 | 4,293 | 1,760 | 0.50x |
| KM-inplace (E) | 1.00 | 30 | 4,319 | 1,200 | 0.50x |

### Robustness (5 seeds each)

| Method | Seeds Correct | Avg DMC | Avg ARD | Avg Floats |
|--------|--------------|---------|---------|------------|
| KM-min (A) | 5/5 | 3,578 | 20 | 1,600 |
| KM-min + verify (D) | 5/5 | 4,293 | 26 | 1,760 |
| KM-inplace (E) | 5/5 | 4,319 | 30 | 1,200 |

All three optimized variants achieve 100% accuracy across all 5 seeds with deterministic DMC values.

## Analysis

### What worked

- **KM-min (1 sample)** reduces DMC by 58% vs GF2 and 83% vs baseline KM. For pure parity, influence is binary (0 or 1), never fractional. A single sample per bit suffices for perfect identification. This cuts total floats from 4,420 to 1,600.
- **KM-inplace** achieves the fewest total floats (1,200) by eliminating the separate x_flip buffer. It reads x twice per iteration (once for original, once for flipped computation) with minimal stack distance (20-40 floats).
- All stack distances in the optimized variants are tiny (20-40 floats), well within L1 cache. Every access is a cache hit.

### What didn't work

- **Shared-buffer KM (B)** produced identical DMC to baseline KM (20,633). Reusing buffer names does not help because the total number of floats read remains the same -- the buffer size (100 floats = 5 samples * 20 bits) resets the clock distance identically each iteration.
- **GF2 fine-grained tracking (C)** was 22x worse than baseline GF2. Fine-grained tracking of Gaussian elimination reveals the true cost: each of n=20 column operations reads the full augmented matrix (21*21 = 441 elements), accumulating massive total floats (17,660) and large stack distances.

### Surprise

The GF2 baseline DMC of 8,607 was artificially low because the harness only tracks `write(A) -> read(A) -> write(solution)`, ignoring the O(n^2) row operations of Gaussian elimination. When tracked properly, GF2's real DMC is 189,056 -- 22x higher. This means KM-min's advantage over GF2 is even larger than the baseline numbers suggest.

The true DMC ranking with honest tracking would be:

| Method | DMC (honest) |
|--------|-------------|
| KM-min | 3,578 |
| KM-inplace | 4,319 |
| GF2 (honest) | 189,056 |
| KM baseline | 20,633 |
| SGD | 1,278,460 |

### Why KM-min wins on DMC

DMC = sum(size * sqrt(stack_distance)). KM-min wins on both factors:

1. **Small buffers**: Each buffer is only 20 floats (1 sample * 20 bits), vs GF2's 420-element matrix.
2. **Tight streaming loop**: Write x (20 floats), immediately read x (distance = 20), process one bit, repeat. Every read has distance exactly 20. Data stays in L1 cache the entire time.

The per-buffer analysis shows KM-min has uniform stack distance of 20 for every single read -- the theoretical minimum for a 20-float buffer that must be written before reading.

## Open Questions

- KM-min uses 1 influence sample, which works perfectly for parity (influence is deterministic). For noisy parity or non-parity functions (AND, threshold), more samples are needed. What is the minimum sample count that maintains accuracy for noisy settings?
- Can we reduce DMC further by processing multiple bits per sample (vectorized influence estimation on a single x vector)?
- The GF2 harness tracking is incomplete (misses row operations). Should the harness be updated to track GF2 more honestly?

## Impact on DISCOVERIES.md

New DMC leader: KM-min achieves DMC 3,578 on n=20/k=3, 58% below the previous best (GF2 at 8,607). Parity influence is deterministic, so 1 sample per bit suffices. GF2's harness-measured DMC was artificially low due to incomplete tracking of Gaussian elimination steps.

Updated rankings:

| Method | ARD | DMC | Total Floats |
|--------|-----|-----|-------------|
| KM-min (1 sample) | 20 | 3,578 | 1,600 |
| KM-inplace | 30 | 4,319 | 1,200 |
| GF2 (harness) | 420 | 8,607 | 860 |
| KM (5 samples) | 92 | 20,633 | 4,420 |

## Files

- Experiment code: `src/sparse_parity/experiments/exp_dmc_optimize.py`
- Results: `results/exp_dmc_optimize/results.json`
- This document: `findings/exp_dmc_optimize.md`

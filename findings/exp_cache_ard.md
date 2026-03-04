# Experiment Cache-ARD: Cache-Aware Memory Tracking

**Date**: 2026-03-04
**Status**: SUCCESS
**Answers**: Open question #2 — "What does ARD look like with a cache model?"

## Hypothesis

If we add LRU cache simulation to MemTracker, then batch-32 will show dramatically higher hit rates than single-sample, because parameters stay resident in cache across the batch.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20 |
| k_sparse | 3 |
| hidden | 200 and 1000 |
| lr | 0.1 |
| wd | 0.01 |
| batch_size | 32 |
| max_epochs | 1 (single step) |
| n_train | 500 |
| seed | 42 |
| method | standard (single-sample vs batch) |

## Results

| Metric | Value |
|--------|-------|
| Best test accuracy | N/A (single step, not training to convergence) |
| Epochs to >90% | N/A |
| Wall time | <1s per comparison |
| CacheTracker implemented | YES |
| Key finding | L2 cache eliminates ALL misses for both methods |

## Key Table

| Hidden | Cache | W1 fits? | Method | Hit Rate | Eff. ARD | Misses | Total Floats |
|--------|-------|----------|--------|----------|----------|--------|--------------|
| 200 | L1 (32KB) | YES | single-sample | 100% | 0 | 0 | 492,731 |
| 200 | L1 (32KB) | YES | batch-32 | 73% | 112,187 | 140 | 364,278 |
| 200 | L2 (256KB) | YES | single-sample | 100% | 0 | 0 | 492,731 |
| 200 | L2 (256KB) | YES | batch-32 | 100% | 0 | 0 | 364,278 |
| 1000 | L1 (32KB) | NO | single-sample | 91% | 33,848 | 49 | 2,455,931 |
| 1000 | L1 (32KB) | NO | batch-32 | 69% | 612,424 | 188 | 2,132,040 |
| 1000 | L2 (256KB) | YES | single-sample | 100% | 0 | 0 | 2,455,931 |
| 1000 | L2 (256KB) | YES | batch-32 | 100% | 0 | 0 | 2,132,040 |

## Analysis

### What worked

- CacheTracker correctly extends MemTracker with LRU simulation
- At L2 (256KB), both methods achieve 100% hit rate — the entire working set fits, confirming exp_b's intuition
- Batch-32 accesses 13% fewer total floats than 32 single-sample steps (2.13M vs 2.46M for hidden=1000), confirming the parameter-traffic reduction found in exp_b
- The cache model gives a clear binary answer: if your cache fits W1, reuse distance is irrelevant

### What didn't work

- Batch does NOT have better L1 cache behavior than single-sample — it's actually worse
  - hidden=200, L1: batch 73% vs single-sample 100%
  - hidden=1000, L1: batch 69% vs single-sample 91%
- The per-sample temporaries in batch (h_pre_0...h_pre_31, h_0...h_31, dh_0, etc.) thrash the L1 cache, evicting parameters that single-sample keeps resident

### Surprise

**Single-sample is more cache-friendly than batch at L1.** Each single-sample step has a tiny working set (~hidden*3 + n_bits floats for temporaries) that fits alongside W1 in L1. Batch creates 32 sets of these temporaries, blowing out the cache. The conventional wisdom "batch reuses parameters" only holds when the cache is large enough to hold both parameters AND all per-sample temporaries.

**The real batch advantage is total traffic, not locality.** Batch-32 does 16x fewer parameter writes (from exp_b) and 13% fewer total float accesses. On a memory-bandwidth-bound system, this is the actual energy saving — not cache hit rate.

## Open Questions (for next experiment)

- Can we get the best of both worlds? Tiled batching: process sub-batches of 4-8 that fit in L1 with their temporaries, accumulate gradients, then do one parameter update
- What's the L1-optimal batch size? Somewhere between 1 and 32 there's a sweet spot where temporaries still fit alongside W1
- Does per-layer + batching change the cache picture? Per-layer processes one layer at a time, potentially fitting more in L1

## Files

- CacheTracker: `src/sparse_parity/cache_tracker.py`
- Experiment: `src/sparse_parity/experiments/exp_cache_ard.py`
- Results: `results/exp_cache_ard/results.json`

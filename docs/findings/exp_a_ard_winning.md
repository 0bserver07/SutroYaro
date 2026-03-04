# Experiment A: ARD on Winning 20-bit Sparse Parity Config

**Date:** 2026-03-04
**Status:** Complete

## Goal

Measure Average Reuse Distance (ARD) on the winning config that solved 20-bit sparse parity (k=3), across all 3 training variants: standard backprop, fused layer-wise, and per-layer forward-backward.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20 |
| k_sparse | 3 |
| hidden | 500 (reduced from 1000 for runtime) |
| LR | 0.1 |
| WD | 0.01 |
| n_train | 500 |
| n_test | 200 |
| seed | 42 |

All 3 variants converged to 96.5% test accuracy in 6 epochs. ARD was measured on a single training step where the hinge margin < 1 (sample 41, margin=0.9889), ensuring the full forward+backward+update path was exercised.

## Results

| Variant | Weighted ARD | Reads | Writes | Total Floats | ARD vs Standard |
|---------|-------------|-------|--------|-------------|-----------------|
| standard | 17,976 | 24 | 18 | 51,073 | baseline |
| fused | 17,741 | 24 | 18 | 51,073 | 1.3% better |
| **perlayer** | **17,299** | **15** | **13** | **47,566** | **3.8% better** |

## Analysis

### Per-layer wins, but modestly

Per-layer forward-backward achieves the best (lowest) ARD at 17,299 floats -- a 3.8% reduction vs standard backprop. The fused variant sits in between at 1.3% better.

### Why the improvement is small

The ARD is dominated by the **W1 buffer** (10,000 floats = hidden x n_bits = 500 x 20). W1 accounts for ~75% of all float reads. Its reuse distance is ~11,042-29,000 regardless of variant, because:

1. W1 is written at the start (initial state)
2. W1 is first read during the forward pass (distance ~11,042)
3. W1 is read again during backward (distance varies by variant)

The time between writing W1 and first reading it is fixed at ~11,042 floats (= all the other buffer writes before forward starts). This dominates the weighted ARD.

### Where per-layer wins

Per-layer shows fewer total operations (15 reads vs 24, 13 writes vs 18) because:
- It avoids separate gradient buffers (dW2, db2, dh, dout, dh_pre)
- Updates are fused inline with backward computation
- W2's reuse distances are tighter: max 14,544 (perlayer) vs 39,570 (standard)

The key per-buffer improvement is **W2**: standard backprop reads W2 three times with avg distance 22,387, while per-layer reads it three times with avg distance 13,877. This is because per-layer updates W2 immediately after computing its gradient, keeping it in "cache" between forward and backward.

### The bottleneck is W1

W1 at 10,000 floats is 20x larger than any other buffer. Any optimization that does not reduce W1's reuse distance will have limited impact on overall ARD. To get meaningful ARD improvement, we need:

1. **Smaller models** -- reduce hidden dimension
2. **Block/tile updates** -- update W1 in chunks so each chunk stays in cache
3. **Forward-Forward** -- Hinton's algorithm avoids backprop entirely, potentially allowing W1 to be read once and updated in-place

## Key Answer

**Does per-layer update still give better ARD on 20-bit?** Yes, but only 3.8% better. The improvement is real but modest because W1 dominates the ARD at this scale. The per-layer variant's main advantage is eliminating intermediate gradient buffers and tightening W2's reuse distance.

## Files

- Experiment: `src/sparse_parity/experiments/exp_a_ard_winning.py`
- Results: `results/exp_a_ard_winning/results.json`

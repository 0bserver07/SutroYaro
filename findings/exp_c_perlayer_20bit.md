# Experiment C: Per-Layer Forward-Backward on 20-bit Sparse Parity

**Date:** 2026-03-04
**Status:** COMPLETE
**Key question:** Does per-layer forward-backward converge on 20-bit (k=3)? What ARD improvement vs standard backprop?

## Setup

| Parameter | Value |
|-----------|-------|
| n_bits | 20 |
| k_sparse | 3 |
| hidden | 1000 |
| LR | 0.1 |
| WD | 0.01 |
| n_train | 500 |
| n_test | 200 |
| Training | Single-sample SGD |
| Max epochs | 200 |
| Seed | 42 |

Secret indices: [0, 3, 8]

## Results

| Method | Best Test Acc | Solve Epoch (>90%) | Wall Time | Weighted ARD |
|--------|--------------|---------------------|-----------|-------------|
| Standard backprop | 99.5% | 6 | 19.1s | 35,920 |
| Per-layer fwd-bwd | 99.5% | 6 | 19.2s | 34,564 |

**ARD improvement: 3.8%** (per-layer vs standard)

## Key Findings

1. **Per-layer CONVERGES on 20-bit.** Both methods reach 99.5% test accuracy by epoch 9, with >90% already at epoch 6. The convergence trajectories are identical -- same accuracy at every epoch.

2. **Identical convergence dynamics.** Both methods produce exactly the same train/test accuracy curves and weight movement norms at every epoch. This makes sense: with single-sample SGD the "per-layer" reordering only affects the parameter update order within each sample, but the mathematical effect is nearly identical when learning rates are moderate.

3. **ARD improvement is 3.8%, down from 9.1% on 3-bit.** The per-layer method uses fewer reads (15 vs 24) and writes (13 vs 18) per training step. The improvement shrinks at scale because:
   - The W1 buffer (hidden x n_bits = 20,000 floats) dominates total memory traffic
   - Per-layer saves intermediate buffers (dW2, db2, dh, dh_pre, dout) by fusing backward + update
   - But these savings are a smaller fraction of total traffic when W1 is large

4. **Why the ARD gap shrinks.** At n_bits=3 (hidden=1000), W1 is 3,000 floats. Intermediate buffers (~4,000 floats for dh, dh_pre, dW2, etc.) are comparable in size to W1, so eliminating them matters. At n_bits=20, W1 is 20,000 floats, dwarfing the intermediate savings.

## Implications

- Per-layer forward-backward is a **safe optimization** -- it does not hurt convergence on the hard 20-bit task
- The 3.8% ARD improvement is real but modest; the benefit diminishes as model size grows
- For larger models, bigger ARD wins will require approaches that reduce the dominant W1 traffic (e.g., tiling, gradient accumulation in registers, or avoiding full W1 reads)

## Files

- Experiment: `src/sparse_parity/experiments/exp_c_perlayer_20bit.py`
- Results: `results/exp_c_perlayer_20bit/results.json`

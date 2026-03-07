# Experiment exp_pebble_game: Pebble Game Optimizer for Sparse Parity

**Date**: 2026-03-06
**Status**: SUCCESS
**Answers**: Can execution reordering (pebble game) reduce energy cost without changing the learning algorithm (SGD)?

## Hypothesis

By modeling one training step as a pebble game on a computation DAG and optimizing the topological execution order, we can reduce total energy cost compared to the standard forward-then-backward ordering. Energy is estimated using a tiered memory model: register (5 pJ), L1 (20 pJ), L2 (100 pJ), HBM (640 pJ) per float access.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20 |
| k_sparse | 3 |
| hidden | 1000 |
| lr | 0.1 |
| wd | 0.01 |
| max_epochs | 30 |
| n_train | 500 |
| n_test | 200 |
| seed | 42 |
| orderings sampled | 5,758 unique |
| L1 capacity | 16,384 floats (64 KB) |
| L2 capacity | 65,536 floats (256 KB) |

## Results

| Strategy | Energy (uJ) | vs Standard | Accuracy | ARD |
|----------|-------------|-------------|----------|-----|
| Standard | 11.00 | baseline | 98% | 45,275 |
| Fused | 10.84 | -1.5% | 57% (BROKEN) | 42,248 |
| Per-layer | 10.84 | -1.5% | 57% (BROKEN) | 43,329 |
| Optimal pebble | 10.76 | -2.2% | 98% | 42,034 |

Energy distribution over 5,758 sampled orderings:

| Statistic | Energy (uJ) |
|-----------|-------------|
| Min | 10.76 |
| Max | 11.08 |
| Mean | 10.92 |
| Median | 10.92 |
| Std | 0.10 |

## Analysis

### What worked

- **Optimal pebbling reduces energy by 2.2% while preserving accuracy**: the best ordering interleaves small-tensor updates (b2, b1) between backward ops, improving cache residency for medium-sized tensors (W2, dW2, h)
- **Optimal ordering delays large-tensor materialization**: dW1 (20,000 floats) and upd_W1 are pushed to the end, keeping the working set smaller during the middle operations
- **The optimal order also achieves the best ARD (42,034)**, validating that pebble game energy and ARD are correlated

### What did not work

- **Fused and per-layer orderings destroy training accuracy (57% = random chance)**: these orderings update W2 before computing `dh = W2^T @ dy`, so the backward pass uses already-updated weights. This is a read-after-write hazard on mutable parameters that the basic DAG dependency model fails to capture
- **The improvement window is very narrow (3% between best and worst)**: because W1 (20,000 floats) always lands in L2 (capacity 65,536) regardless of ordering, and W1 accesses dominate total energy

### Key insight: DAG dependencies are insufficient for mutable state

The pebble game DAG captures data-flow dependencies (which tensors must exist before an op runs) but does not capture read-after-write hazards on shared mutable tensors. The `upd_W2` operation modifies W2 in place, but `bwd_dh` reads W2 for backpropagation. The DAG shows no dependency between them because they are on different branches. A correct model must add "anti-dependencies": if op A reads a tensor that op B writes (and A needs the pre-write value), A must precede B.

### Energy breakdown by tier

| Tier | Standard | Optimal | Notes |
|------|----------|---------|-------|
| Register | 0.00 uJ (0.0%) | 0.00 uJ (0.0%) | Only scalar tensors (loss, dy, b2) |
| L1 | 0.30 uJ (2.7%) | 0.36 uJ (3.3%) | h, h_pre, dh, dW2 when recently accessed |
| L2 | 10.70 uJ (97.3%) | 10.40 uJ (96.7%) | W1 (20K floats) dominates |
| HBM | 0.00 uJ (0.0%) | 0.00 uJ (0.0%) | Total working set fits in L2 |

The optimal ordering shifts ~0.3 uJ from L2 to L1 by keeping medium-sized tensors (1,000 floats each) more recently accessed when they are needed.

### Optimal ordering (the pebble schedule)

```
 1. fwd_linear1    (read W1, x, b1 -> write h_pre)
 2. fwd_relu       (read h_pre -> write h)
 3. fwd_linear2    (read W2, h, b2 -> write y_hat)
 4. fwd_loss       (read y_hat, y -> write loss)
 5. bwd_loss       (read loss, y_hat, y -> write dy)
 6. bwd_dW2        (read dy, h -> write dW2)
 7. bwd_db2        (read dy -> write db2)
 8. upd_b2         (read b2, db2 -> write b2_new)    <-- small update early
 9. bwd_dh         (read W2, dy -> write dh)          <-- W2 still original
10. bwd_relu       (read dh, h_pre -> write dh_pre)
11. upd_W2         (read W2, dW2 -> write W2_new)    <-- W2 update after bwd_dh
12. bwd_db1        (read dh_pre -> write db1)
13. upd_b1         (read b1, db1 -> write b1_new)
14. bwd_dW1        (read dh_pre, x -> write dW1)     <-- large tensor last
15. upd_W1         (read W1, dW1 -> write W1_new)    <-- largest op last
```

The key moves vs standard ordering: (a) upd_b2 is pulled forward to step 8, (b) upd_W2 is delayed until after bwd_dh uses W2, (c) bwd_dW1 and upd_W1 (the two 20K-float operations) are pushed to the very end.

## Open Questions

- With larger models (deeper networks), does the improvement from pebble game optimization grow? The working set may exceed L2, putting some tensors in HBM where the cost difference is 6.4x
- Can anti-dependencies (read-after-write on mutable params) be encoded into the DAG automatically to prevent invalid orderings like fused/perlayer?
- Would activation checkpointing (recomputation) trade energy for memory and open up more optimization opportunities?
- For batch training (multiple samples), the pebble game becomes more complex -- does the optimal ordering change?

## Files

- Experiment: `src/sparse_parity/experiments/exp_pebble_game.py`
- Results: `results/exp_pebble_game/results.json`

# Experiment exp_tiled_w1: Tiled W1 Updates for Sparse Parity

**Date**: 2026-03-06
**Status**: FAILED (ARD increased instead of decreased)

## Hypothesis

W1 (input-to-hidden weights) dominates ARD at 75% of all float reads. W1 is 20x1000 = 20,000 floats = 80KB, which exceeds L1 cache (64KB). Splitting W1 into tiles along the hidden dimension (e.g., T=250 -> 5,000 floats = 20KB per tile) and processing each tile's forward/backward/update before moving to the next should keep each tile in L1 cache, reducing reuse distance.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20 |
| k_sparse | 3 |
| hidden (ARD) | 1000 |
| hidden (accuracy) | 500 |
| tile_sizes | 50, 100, 250, 500 |
| lr | 0.1 |
| wd | 0.01 |
| n_train | 500 |
| n_test | 200 |
| seeds | 42, 43, 44 |

## Results

### ARD Measurement (hidden=1000, 1 step)

| Method | Avg ARD | ARD Change | Tile KB | N Tiles |
|--------|---------|------------|---------|---------|
| Baseline | 30,759 | --- | N/A | 1 |
| Tiled T=50 | 34,492 | +12.1% | 3.9 | 20 |
| Tiled T=100 | 34,236 | +11.3% | 7.8 | 10 |
| Tiled T=250 | 33,690 | +9.5% | 19.5 | 4 |
| Tiled T=500 | 32,853 | +6.8% | 39.1 | 2 |

### Accuracy Verification (hidden=500, max_epochs=50)

| Method | Avg Acc | Avg Epochs |
|--------|---------|------------|
| Baseline | 99.8% | 23 |
| Tiled T=50 | 99.8% | 23 |
| Tiled T=100 | 99.8% | 23 |
| Tiled T=250 | 99.8% | 23 |
| Tiled T=500 | 99.8% | 23 |

## Analysis

### What happened

Tiling W1 **increased** ARD by 6.8%-12.1% instead of decreasing it. The core problem is structural:

1. **Forward pass cannot be fully tiled**: The output layer needs the full `h` vector (all hidden units), so all tiles must complete their forward pass before the output layer can run. This means tile 0's W1 slice is read in forward, then tiles 1-19 all run their forward passes, then the output layer runs, then backward starts. The W1_tile0 forward-to-backward distance is *longer* than in baseline because the tiled forward is not more compact -- it is the same total computation, just reorganized.

2. **Additional x reads**: In standard backprop, `x` is read once in forward and once in backward (2 reads total). In tiled mode, `x` is read once per tile in forward and once per tile in backward (2*N_tiles reads). This extra reading adds to total floats accessed and increases distances for other buffers.

3. **The MemTracker measures software-level reuse distance**: It counts intervening float accesses between write and read of a buffer. Tiling does not reduce the amount of computation between W1_tile_k's forward read and backward read -- the output layer and backward output layer still intervene. The only way tiling helps is at the hardware cache level (L1/L2), which the MemTracker does not simulate.

### What did work

- **Accuracy is perfectly maintained**: Tiling does not change the mathematics of backprop. All tile sizes produce identical accuracy and convergence speed as baseline.
- **Smaller tiles trend toward higher ARD**: More tiles = more overhead from repeated `x` reads and more buffer fragmentation.

### Key insight

The MemTracker ARD metric measures **software-level reuse distance** (intervening float accesses between producer and consumer). Tiling is a **hardware-level cache optimization** that works by ensuring a W1 tile fits in L1 so it remains cached between forward and backward. These are fundamentally different metrics:

- Software ARD: determined by the total computation between forward and backward. Tiling cannot reduce this.
- Hardware cache hit rate: determined by whether a buffer fits in L1/L2. Tiling helps here because a 20KB tile fits in 64KB L1, while the full 80KB W1 does not.

To measure the hardware benefit, one would need the CacheTracker (LRU cache simulation) rather than the MemTracker.

## Open Questions

- Would CacheTracker (from `cache_tracker.py`) show the expected L1 hit rate improvement for tiled W1?
- Can we design a "fully tiled" approach where each tile's backward runs immediately after its forward, before processing the next tile? This would require approximating the output layer or using a different loss formulation.
- Is there a tiling strategy that also tiles the output layer (W2) to keep h_tile in cache?

## Files

- Experiment: `src/sparse_parity/experiments/exp_tiled_w1.py`
- Results: `results/exp_tiled_w1/results.json`

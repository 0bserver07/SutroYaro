# Experiment exp_binary_weights: Binary Weights for Sparse Parity

**Date**: 2026-03-06
**Status**: PARTIAL SUCCESS (n=3 solved, n=20 failed)
**Approach**: #13 -- BinaryConnect (Courbariaux et al. 2015)

## Hypothesis

If we train a 2-layer network with binary weights (+1/-1) and sign activation, sparse parity should be solvable because parity is fundamentally a binary operation. Binary weights reduce inference memory by 13x and energy per step by 1.5x. The straight-through estimator (STE) passes gradients through the sign function unchanged for training.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 3, 20 |
| k_sparse | 3 |
| hidden | 100 (n=3), 200 and 1000 (n=20) |
| lr | 0.01, 0.1 |
| wd | 0.01 |
| batch_size | 32 |
| max_epochs | 200 |
| n_train | 100 (n=3), 1000 (n=20) |
| n_test | 100 (n=3), 500 (n=20) |
| seeds | 42, 43, 44 |
| method | BinaryConnect (sign weights + STE) vs float32 baseline |

## Results

| Config | Method | Avg Acc | Solved | Avg Epochs | Avg Time |
|--------|--------|---------|--------|------------|----------|
| n=3, k=3, h=100 | binary | 100% | 3/3 | 1 | 0.00s |
| n=3, k=3, h=100 | float32 | 100% | 3/3 | 68 | 0.02s |
| n=20, k=3, h=200, lr=0.1 | binary | 55.5% | 0/3 | 200 | 1.51s |
| n=20, k=3, h=200, lr=0.1 | float32 | 100% | 3/3 | 39 | 0.16s |
| n=20, k=3, h=1000, lr=0.01 | binary | 59.7% | 0/3 | 200 | 5.29s |
| n=20, k=3, h=1000, lr=0.01 | float32 | 57.6% | 0/3 | 200 | 1.90s |
| n=20, k=3, h=1000, lr=0.1 | binary | 54.9% | 0/3 | 200 | 5.01s |
| n=20, k=3, h=1000, lr=0.1 | float32 | 100% | 3/3 | 51 | 0.39s |

## Energy and Memory Comparison (per-step, first mini-batch)

| Config | Binary ARD | Float32 ARD | Binary Energy Proxy | Float32 Energy Proxy | Ratio |
|--------|-----------|-------------|--------------------|--------------------|-------|
| n=3, k=3 | 4,667 | 7,679 | 150M | 242M | 1.6x |
| n=20, k=3, h=200 | 13,726 | 19,103 | 987M | 1,602M | 1.6x |
| n=20, k=3, h=1000 | 65,811 | 91,467 | 23.9B | 36.5B | 1.5x |

| Config | Float32 Inference | Binary Inference | Reduction |
|--------|------------------|-----------------|-----------|
| n=3, k=3 | 2.0 KB | 0.4 KB | 4.4x |
| n=20, k=3 (h=1000) | 85.9 KB | 6.5 KB | 13.3x |

## Analysis

### What worked

- **n=3/k=3 solved in 1 epoch by binary weights** vs 68 epochs for float32. When all input bits are parity bits, the sign activation aligns perfectly with the structure: binary weights + sign activation can represent 3-bit parity exactly.
- **Energy per step is 1.5-1.6x lower** for binary due to smaller weight reads (counted as 1/32 of float reads).
- **Inference memory is 13x smaller** for n=20/k=3 with h=1000 (6.5 KB vs 85.9 KB).

### What did not work

- **Binary weights completely fail on n=20/k=3**: ~55% accuracy (near random chance) across all learning rates and hidden sizes. The float32 baseline solves the same configs at 100%.
- **The straight-through estimator is too crude**: The sign function has zero derivative everywhere except at zero. STE approximates this as identity (gradient = 1), which is a poor approximation. For n=20, the network needs to learn which 3 of 20 input bits matter -- this requires precise gradient information that STE destroys.
- **Sign activation kills the gradient signal**: Unlike ReLU (which passes gradients for positive inputs), sign activation with STE passes all gradients unchanged. This removes the selective gating that helps ReLU networks learn feature selection in sparse parity.

### Root cause of failure

The fundamental issue is that BinaryConnect was designed for classification tasks where float32 networks already work and binarization is a compression technique. For sparse parity, the learning problem itself (feature selection) requires gradient precision. The STE approximation is adequate for MNIST/CIFAR where the loss landscape is smooth, but sparse parity has a sharp phase transition that requires precise gradient accumulation to navigate.

The n=3/k=3 success is misleading -- when all bits are relevant, there is no feature selection problem. The network just needs to learn the parity function, which binary weights represent exactly.

### Surprise

- **Binary weights solve n=3/k=3 in a single epoch** -- 80x faster than float32. This confirms that parity IS fundamentally a binary operation, but only when the feature selection problem is trivial.
- **The float32 baseline with h=1000, lr=0.01 also fails** (57.6%), showing that the learning rate matters more than the weight precision for sparse parity convergence.

## Open Questions (for next experiment)

- Can a hybrid approach work? Use float32 for the first layer (feature selection) and binary for the second layer (parity computation).
- Would a better gradient estimator (e.g., clipped STE, or the Gumbel-softmax trick) help binary weights learn feature selection?
- Can binary weights succeed if we pre-select the relevant features (e.g., via LASSO or mutual information) and only need to learn the parity function on the selected features?

## Files

- Experiment: `src/sparse_parity/experiments/exp_binary_weights.py`
- Results: `results/exp_binary_weights/results.json`

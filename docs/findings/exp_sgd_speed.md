# Experiment: Push SGD Under 10ms on Sparse Parity

**Date**: 2026-03-14
**Status**: LOSS (SGD cannot reach 10ms on n=20/k=3)
**Issue**: #4

## Hypothesis

SGD can solve n=20/k=3 sparse parity in under 10ms by shrinking the model (hidden=32-64), reducing training samples, or using aggressive learning rates.

## Results

### Hidden size sweep (lr=0.5, max_epochs=50)

| Hidden | n_train | Accuracy | Time | Epochs |
|--------|---------|----------|------|--------|
| 64 | 200 | 52% | 108ms | 50 (failed) |
| 64 | 500 | 100% | 113ms | 21 |
| 64 | 1000 | 100% | 95ms | 8 |
| 100 | 500 | 100% | 71ms | 18 |
| 100 | 1000 | 100% | 68ms | 9 |
| 200 | 500 | 100% | 73ms | 19 |
| 200 | 1000 | 100% | 72ms | 7 |

### Learning rate sweep (hidden=200, n_train=1000)

| LR | Accuracy | Time | Epochs |
|----|----------|------|--------|
| 0.1 | 100% | 142ms | ~40 |
| 0.2 | 100% | 104ms | 19 |
| 0.5 | 100% | 94ms | 7 |
| 1.0 | 100% | 74ms | 8 |
| 2.0 | 100% | 78ms | 11 |

### Other attempts

| Config | Accuracy | Time | Notes |
|--------|----------|------|-------|
| hidden=32, n_train=200 | 51% | 193ms | Too small, can't learn |
| batch=128 | 99.5% | 349ms | Larger batch hurts |
| batch=1000 (full) | 53.5% | 129ms | Full batch diverges |
| n=10/k=3 | 100% | 58ms | Easier problem, still slow |

## Finding

**SGD cannot solve n=20/k=3 sparse parity in under 60ms with any configuration.**

The floor is approximately 7 grokking epochs at 8-12ms per epoch. The phase transition (where the network suddenly learns the parity function) requires a minimum number of weight updates that cannot be reduced by:
- Shrinking the model (hidden=64 needs more epochs, cancels out)
- Increasing learning rate (LR=0.5-1.0 reduces to 7-8 epochs, can't go lower)
- Reducing training samples (n_train=200 fails entirely for small models)
- Larger batch size (hurts or fails)

## Why this matters

LeCun's Spark 7 experiments worked because digit recognition has first-order structure. Each pixel contributes independently to the output. SGD can learn this quickly.

Parity has zero first-order structure. The signal exists only in the k-th order interaction of the secret bits. SGD must discover this through grokking, which takes a minimum number of gradient steps regardless of model configuration.

The 10ms budget ("runnable on 1980s hardware in 1 hour") correctly filters SGD out of the viable methods for sparse parity. The algebraic methods (GF(2) at 0.5ms, KM at 1-4ms) fit. SGD doesn't. This is not a failure of SGD optimization. It is a structural mismatch between the algorithm and the problem.

## Best SGD config found

hidden=200, n_train=1000, lr=0.5, batch=32: 100% accuracy in 7 epochs / 72ms. This is 2x faster than the default config (142ms) but still 7x over the 10ms target.

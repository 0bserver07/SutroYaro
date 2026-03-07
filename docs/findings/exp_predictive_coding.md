# Experiment exp_predictive_coding: Predictive Coding for Sparse Parity

**Date**: 2026-03-06
**Status**: FAILED (accuracy at chance; ARD worse than backprop)

## Hypothesis

Predictive coding's local learning rule (each layer only talks to its neighbors) should yield smaller ARD than backprop, because the working set per layer update is just its own weights + activations + one neighbor's prediction error. Under certain conditions, predictive coding approximates backprop's weight updates (Millidge et al. 2021), so accuracy should be comparable.

## Config

| Parameter | n=3/k=3 (sanity) | n=20/k=3 (main) |
|-----------|-------------------|------------------|
| hidden | 100 | 1000 |
| lr | 0.01 | 0.1 |
| wd | 0.001 | 0.01 |
| inference_iters | 15 | 15 |
| inf_lr | 0.1 | 0.05 |
| max_epochs | 100 | 50 |
| n_train | 50 | 500 |
| n_test | 50 | 200 |
| seeds | 42, 43, 44 | 42, 43, 44 |

## Results

| Method | Config | Avg Test Acc | Weighted ARD | Converge Epoch |
|--------|--------|-------------|-------------|----------------|
| Predictive Coding | n=3/k=3 | 0.553 | 8,441 | never |
| Backprop | n=3/k=3 | 1.000 | 447 | 2-3 |
| Predictive Coding | n=20/k=3 | 0.512 | 370,005 | never |
| Backprop | n=20/k=3 | 0.998 | 20,438 | 5-6 |

### ARD Comparison

| Config | Backprop ARD | PC ARD | Ratio (BP/PC) | Winner |
|--------|-------------|--------|---------------|--------|
| n=3/k=3 | 447 | 8,441 | 0.05x | Backprop (18.9x lower) |
| n=20/k=3 | 20,438 | 370,005 | 0.06x | Backprop (18.1x lower) |

## Analysis

### What worked

- **MemTracker instrumentation is clean**: the per-buffer breakdown shows exactly where the memory traffic goes. Local buffers (e1, e0, e2, mu1) have small reuse distances (~100-5000 floats), confirming that the error signals are indeed local.
- **Backprop baseline is solid**: 100% accuracy on n=3/k=3 in 2-3 epochs, 99.8% on n=20/k=3 in 5-6 epochs, with ARD ~20,438 (consistent with the baseline ~17,976).

### What didn't work

- **Accuracy is at chance level (~50%)**: Predictive coding completely failed to learn sparse parity. Not even the n=3/k=3 sanity check solved. This is not a hyperparameter issue -- the network genuinely does not learn.
- **ARD is 18x WORSE than backprop, not better**: The 15 inference iterations per sample mean the weight matrices (W1: 20,000 floats, W2: 1,000 floats) are read 32 times each instead of backprop's 2 times (forward + backward). This dominates the ARD calculation. Each full pass through W1 pushes all other buffers far away in the reuse timeline.
- **The "local learning" advantage is an illusion for this architecture**: While individual error signals (e0, e1, e2) have small reuse distances, computing them still requires reading the full weight matrices. The working set per layer is W_l (which is the largest buffer), not just the errors.
- **Pure Python is too slow**: n=20/k=3 with hidden=1000 hit the 120s timeout after only 4-5 epochs. The O(n_inference_iters * hidden * n_input) inner loop is ~300M operations per epoch.

### Surprise

- **The ARD ratio (~18x) exactly matches the inference iteration count (15) + weight update reads (2-3 extra)**. This makes sense: backprop reads each weight matrix ~2 times total (forward + backward). PC reads it ~32 times (2 per inference iteration x 15 iterations + a few more for weight updates). The "local" error signals are cheap, but they require re-reading the expensive weight matrices.
- **Predictive coding's locality benefit only materializes when the network has many layers**: with a 2-layer network, the "only talk to neighbors" constraint doesn't help because every layer IS a neighbor. Backprop in a 2-layer net already has perfect locality -- layer 2 gradients feed directly into layer 1. PC would need 10+ layers to show an advantage, where backprop must store intermediate activations across many layers.
- **The convergence failure is likely fundamental for parity**: PC's generative model (predicting x from h) struggles with XOR/parity because the generative direction is harder than the discriminative direction. The product of k bits is easy to compute forward but hard to invert.

## Open Questions

- Would PC work with many more layers (10+) where backprop's activation storage becomes the bottleneck?
- Would a discriminative PC variant (not generative) fare better on parity?
- Can the inference iteration count be reduced (e.g., 3-5) while maintaining the approximation-to-backprop property?
- Would amortized inference (learning the recognition weights separately) improve convergence?

## Files

- Experiment: `src/sparse_parity/experiments/exp_predictive_coding.py`
- Results: `results/exp_predictive_coding/results.json`
- Findings: `findings/exp_predictive_coding.md`

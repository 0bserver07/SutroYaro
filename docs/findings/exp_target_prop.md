# Experiment exp_target_prop: Target Propagation for Sparse Parity

**Date**: 2026-03-06
**Status**: FAILED
**Approach**: #10 - Target Propagation (Bengio 2014, Lee et al. 2015)

## Hypothesis

Target propagation replaces backprop's chain rule with approximate inverse mappings. Each layer receives a local target computed by inverse functions from the layer above, then trains to match that target. Since weight updates are local after the target computation pass, this should have favorable memory access patterns (lower ARD). The inverse mappings should carry global loss information better than Forward-Forward's purely greedy layer-wise objectives.

## Config

| Parameter | n=3/k=3 (sanity) | n=20/k=3 (main) |
|-----------|-------------------|------------------|
| hidden | 100 | 1000 |
| lr_forward | 0.005 | 0.01 |
| lr_inverse | 0.005 | 0.01 |
| n_train | 50 | 500 |
| n_test | 50 | 200 |
| max_epochs | 500 | 200 |
| grad_clip | 1.0 | 1.0 |
| seeds | 42, 43, 44 | 42, 43, 44 |

## Results

### Accuracy

| Config | Method | Seed 42 | Seed 43 | Seed 44 | Mean |
|--------|--------|---------|---------|---------|------|
| n=3/k=3 | TargetProp | 0.780 | 0.800 | 0.760 | 0.780 |
| n=3/k=3 | Backprop | 0.860 | -- | -- | 0.860 |
| n=20/k=3 | TargetProp | 0.555 | 0.530 | 0.550 | 0.545 |
| n=20/k=3 | Backprop | 0.585 | -- | -- | 0.585 |

Neither method solves n=20/k=3 (both near chance ~50%). Both overfit on n=3/k=3 training data but generalization is limited (test caps around 78-86%).

### ARD (single-step memory reuse distance)

| Config | TargetProp ARD | Backprop ARD | Ratio (BP/TP) |
|--------|---------------|--------------|---------------|
| n=3/k=3 | 1,323 | 842 | 0.64x |
| n=20/k=3 | 37,788 | 33,991 | 0.90x |

TargetProp has **higher** ARD than backprop (1.57x for 3-bit, 1.11x for 20-bit). This is the opposite of what was expected.

### Memory Access Details (n=20/k=3, single step)

| Buffer | TP Reads | TP Avg Dist | BP Reads | BP Avg Dist |
|--------|----------|-------------|----------|-------------|
| W1 (20k) | 2 | 42,555 | 2 | 38,554 |
| h1 (1k) | 4 | 8,503 | 2 | 2,002 |
| W2 (1k) | 2 | 31,044 | 2 | 26,544 |
| G2 (1k) | 2 | 27,543 | -- | -- |
| g2_bias (1k) | 2 | 27,543 | -- | -- |

TargetProp total floats accessed: 103,068 vs Backprop: 97,067 (6% more).

## Analysis

### What did not work

- **Target propagation failed to solve sparse parity at n=20/k=3.** Test accuracy plateaus at ~53% (near random chance). The hidden loss L_hid converges to near zero, meaning the hidden layer perfectly matches its target -- but the targets themselves are not useful. The inverse mapping G2 is a simple linear map from R^1 to R^1000, which cannot provide meaningful targets for learning a nonlinear function like parity.

- **The fundamental mismatch**: sparse parity requires the network to learn XOR-like interactions between specific input bits. A linear inverse G2 mapping a scalar label (+1/-1) to a 1000-dimensional hidden target cannot encode which specific input bits matter. It produces essentially the same target for all +1 inputs and the same target for all -1 inputs, regardless of the actual input pattern.

- **ARD is worse, not better.** Target propagation requires extra buffers (G2, g2_bias, t1) and extra reads of h1 (4 reads vs 2 for backprop). The inverse training step adds memory accesses that increase total reuse distance. This contradicts the hypothesis that local updates would reduce ARD.

- **Gradient clipping was necessary** to prevent NaN divergence but severely limits learning capacity. Without clipping, lr=0.1 causes immediate overflow due to the MSE loss scaling with hidden dimension (loss ~ hidden * error^2).

### Why it failed

1. **Linear inverse is too weak**: The inverse g2 maps R^1 -> R^hidden linearly. For sparse parity, different inputs with the same label need different hidden representations, but a linear inverse produces only two possible targets (one for +1, one for -1). This is fundamentally insufficient.

2. **Target collapse**: The hidden target t1 = G2 @ y + g2_bias gives the same target for all samples with y=+1 and a different fixed target for y=-1. Layer 1 then tries to produce these fixed targets regardless of input, destroying any input-dependent learning.

3. **The inverse loss converges trivially**: L_inv approaches 0 because G2 learns to map y_hat back to h1 on average, but this doesn't mean the targets are useful for classification.

### Comparison with Forward-Forward (exp_e)

Both Forward-Forward and Target Propagation are local learning alternatives to backprop, but they fail differently on sparse parity:
- Forward-Forward (exp_e): stuck at 50% on n=3/k=3 (never learns). Uses goodness-based objective which is wrong for parity.
- Target Propagation: reaches ~78% on n=3/k=3 (partial learning) but ~53% on n=20/k=3. The inverse mapping bottleneck prevents scaling.
- Standard backprop (this experiment): reaches 86% test on n=3/k=3 but also only 58% on n=20/k=3 with MSE loss and small lr.

## Open Questions

- Would a **nonlinear inverse** (e.g., a small MLP for g2) fix the target collapse problem?
- Would **difference target propagation** (Lee et al. 2015), which uses t1 = h1 + g2(y) - g2(y_hat), avoid the fixed-target problem?
- Is sparse parity fundamentally incompatible with target propagation because the label-to-hidden mapping is many-to-one (exponentially many inputs map to each label)?
- Would combining target propagation with the hinge loss from the reference backprop implementation (instead of MSE) improve convergence?

## Files

- Experiment: `src/sparse_parity/experiments/exp_target_prop.py`
- Results: `results/exp_target_prop/results.json`

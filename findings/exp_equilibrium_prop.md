# Experiment exp_equilibrium_prop: Equilibrium Propagation for Sparse Parity

**Date**: 2026-03-06
**Status**: FAILED
**Approach**: #9 -- Scellier & Bengio (2017) Equilibrium Propagation

## Hypothesis

Equilibrium propagation can solve sparse parity without backpropagation by using two forward relaxation phases (free and clamped). The weight update rule dW = (1/beta) * (s_clamped - s_free) approximates backprop gradients in the limit of small beta. Since EP uses only forward passes, it should have fundamentally different ARD characteristics than backprop.

## Config

| Parameter | n=3/k=3 | n=20/k=3 |
|-----------|---------|----------|
| n_bits | 3 | 20 |
| k_sparse | 3 | 3 |
| hidden | 100 | 1000 |
| n_train | 50 | 500 |
| n_test | 50 | 200 |
| lr | 0.1 | 0.05 |
| beta | 0.2 | 0.2 |
| free_steps | 30 | 30 |
| clamp_steps | 30 | 30 |
| step_size | 0.5 | 0.5 |
| max_epochs | 200 | 100 |
| seeds | 42, 43, 44 | 42, 43, 44 |

## Results

| Config | Seed | Best Train | Best Test | Converge Epoch | Time | Weighted ARD |
|--------|------|-----------|-----------|----------------|------|-------------|
| n=3, k=3 | 42 | 0.500 | 0.440 | - | 8.3s | 18,121 |
| n=3, k=3 | 43 | 0.620 | 0.820 | - | 8.3s | 18,121 |
| n=3, k=3 | 44 | 0.620 | 0.740 | - | 8.3s | 18,121 |
| n=20, k=3 | 42 | 0.528 | 0.525 | - | 91.9s | 711,003 |
| n=20, k=3 | 43 | 0.588 | 0.555 | - | 95.1s | 711,003 |
| n=20, k=3 | 44 | 0.768 | 0.745 | - | 94.4s | 711,003 |

### Averages

| Config | Avg Train | Avg Test | Avg Time | Avg ARD |
|--------|-----------|----------|----------|---------|
| n=3, k=3 | 0.580 | 0.667 | 8.3s | 18,121 |
| n=20, k=3 | 0.628 | 0.608 | 93.8s | 711,003 |

## Analysis

### What worked

- **The algorithm runs and produces gradient updates** -- the EP framework is correctly implemented with free and clamped phases, and weights do update.
- **Some seeds show partial learning** -- seed 43 on n=3/k=3 reached 82% test accuracy, and seed 44 on n=20/k=3 reached 74.5% test accuracy, both well above the 50% chance baseline.
- **ARD is well-defined** -- the two-phase relaxation process has a clear memory access pattern: 401 reads, 134 writes per training step (for n=3/k=3 with 30 relaxation steps per phase).

### What didn't work

- **Failed to converge on any config** -- no seed reached 90% test accuracy on either n=3/k=3 or n=20/k=3. The network gets stuck in local minima where the output saturates to a constant prediction.
- **Tanh saturation trap** -- the network quickly saturates (cost hits a plateau like 1.0 or 0.76 and stays there for hundreds of epochs). Once the output node saturates, the EP gradient signal vanishes because d(tanh)/d(pre) approaches 0.
- **Very slow per epoch** -- each epoch requires 2 * n_steps forward relaxation iterations per sample. For n=20/k=3 with 500 training samples and 30 steps per phase, that is 30,000 forward computations per epoch, taking ~0.9s per epoch. Backprop baseline solves this in ~5 epochs (0.12s total).
- **No grokking observed** -- unlike SGD which can grok after many epochs, EP shows no sign of delayed generalization. The loss plateau is a hard wall.

### Surprise

- **EP struggles even on n=3/k=3 (full parity)** -- this is the easiest possible config where all bits matter, yet EP cannot reliably solve it. SGD solves this trivially in 1-3 epochs. This suggests the EP relaxation dynamics are fundamentally mismatched with the parity function's XOR-like structure.
- **Seed-dependent partial learning** -- seed 44 on n=20/k=3 reached 76.8% train accuracy while seed 42 was stuck at 52.8%. The initial random weights determine whether the network finds any useful features at all, suggesting a very rough loss landscape.
- **Compute cost is extreme** -- n=20/k=3 took 93.8s average per seed for 100 epochs (total ~281s across 3 seeds). SGD baseline solves it in 0.12s. That is a ~2,300x slowdown for a worse result.

## Comparison with Backprop and Forward-Forward

| Method | n=3/k=3 Test | n=20/k=3 Test | Per-step ARD (n=3) |
|--------|-------------|--------------|-------------------|
| SGD (backprop) | 1.000 | 1.000 | ~4,000 (est.) |
| Forward-Forward | 0.500-0.660 | 0.500-0.515 | ~2,900 |
| Equilibrium Prop | 0.440-0.820 | 0.525-0.745 | 18,121 |

EP has higher ARD than both backprop and Forward-Forward because the iterative relaxation requires repeatedly reading the full weight matrices (30 times per phase, 60 total per training step).

## Open Questions

- Would a continuous Hopfield network formulation with modern Hopfield energy work better?
- Can the saturation problem be fixed with layer normalization or different activation functions?
- Would a contrastive Hebbian learning variant (a predecessor of EP) perform differently?
- Is the fundamental issue that parity requires precise cancellation, which iterative relaxation cannot achieve?

## Files

- Experiment: `src/sparse_parity/experiments/exp_equilibrium_prop.py`
- Results: `results/exp_equilibrium_prop/results.json`

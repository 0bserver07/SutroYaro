# Experiment exp_mutual_info: Mutual Information Estimation for Sparse Parity

**Date**: 2026-03-06
**Status**: SUCCESS
**Answers**: Blank slate approach -- can MI between parity products and labels identify the secret subset?

## Hypothesis

For sparse parity with secret S, MI(y; prod(x_S)) should equal log(2) ~ 0.693 nats (1 bit) for the true subset, and ~0 for all wrong subsets. Unlike Fourier/Walsh-Hadamard (which computes exact correlation), MI detects any nonlinear relationship via a 2x2 contingency table. For binary parity, both approaches should work, but MI generalizes to non-binary settings.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 50 |
| k_sparse | 3, 5 |
| method | exhaustive MI over all C(n,k) subsets |
| MI estimator | 2x2 contingency table (exact for binary {-1,+1}) |
| n_samples | 500 |
| seeds | 42, 43, 44 |

## Results

| Config | C(n,k) | Avg MI (true) | Avg MI (wrong) | Avg MI gap | Avg time | Correct |
|--------|--------|---------------|----------------|------------|----------|---------|
| n=20, k=3 | 1,140 | 0.6925 | 0.0010 | 0.6799 | 0.033s | 3/3 |
| n=50, k=3 | 19,600 | 0.6922 | 0.0010 | 0.6731 | 0.569s | 3/3 |
| n=20, k=5 | 15,504 | 0.6915 | 0.0010 | 0.6756 | 0.453s | 3/3 |

## Key Table

| Method | n=20/k=3 | n=50/k=3 | n=20/k=5 |
|--------|----------|----------|----------|
| MI (this exp) | 0.033s, 100% | 0.569s, 100% | 0.453s, 100% |
| Fourier (exp_fourier) | 0.009s, 100% | 0.13s, 100% | 1.2s, 100% |
| Random search (exp_evolutionary) | 0.011s, 100% | 0.142s, 100% | 0.426s, 100% |
| SGD (baseline) | 0.12s, 100% | FAIL (54%) | 14 epochs (n=5000) |
| SGD + curriculum | -- | 20 epochs, >90% | -- |

## Analysis

### What worked

- **MI identifies the true subset with 100% accuracy on all configs and all seeds.** The true subset's MI is always within 0.002 nats of the theoretical maximum log(2) = 0.693 nats.
- **The MI gap is enormous**: best MI ~ 0.69 vs next-best ~ 0.01-0.02. The signal-to-noise ratio is ~40-70x, making identification trivial.
- **MI solves n=50/k=3 in 0.57s**, a config that SGD fails on (54% accuracy) without curriculum learning.
- **Sample complexity is low**: 20 samples suffice for n=20/k=3 (3/3 correct). The MI gap grows with more samples (0.35 at 20 samples, 0.68 at 500 samples) but even small gaps are enough for correct identification.

### What didn't work

- **MI is ~3.7x slower than Fourier on n=20/k=3** (0.033s vs 0.009s) and **~4.4x slower on n=50/k=3** (0.569s vs 0.13s). The contingency table computation (looping over 2x2 cells, computing boolean masks) is heavier than Fourier's single mean(y * parity) operation.
- **MI is faster than Fourier on n=20/k=5** (0.453s vs 1.2s). This appears to be due to implementation differences rather than algorithmic advantage; both are O(C(n,k) * n_samples).
- **ARD is identical to Fourier** (1,147,375 vs 17,976 for SGD). Both methods stream over the entire dataset for each subset with no weight reuse.

### Surprise

- **MI and Fourier give identical ARD** (1,147,375). This makes sense: both iterate over all C(n,k) subsets and read x and y for each one. The memory access pattern is the same streaming pattern regardless of whether you compute correlation or MI.
- **Wrong subsets have MI extremely close to 0** (mean ~0.001 nats). For independent binary variables with 500 samples, the finite-sample MI bias is about 1/(2 * n_samples * ln(2)) ~ 0.001, which matches perfectly. MI correctly reports near-zero for independent variables.
- **MI provides no advantage over Fourier for binary parity.** Both detect the same signal (perfect correlation/dependence between parity product and label). MI's generality (detecting non-linear relationships) is unnecessary here because the relationship IS the linear product.

## Open Questions

- Does MI outperform Fourier on a modified parity problem where the label is a nonlinear function of the parity (e.g., y = sign(prod(x_S) + noise))? MI should be more robust to such corruptions.
- Can kNN-based MI estimators (Kraskov) work on continuous-valued features without the discretization step?
- For large k, can MI be computed on partial subsets to prune the search space?

## Files

- Experiment: `src/sparse_parity/experiments/exp_mutual_info.py`
- Results: `results/exp_mutual_info/results.json`

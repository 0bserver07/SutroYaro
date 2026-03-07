# Experiment exp_lasso: LASSO on Interaction Features for Sparse Parity

**Date**: 2026-03-06
**Status**: SUCCESS
**Answers**: Can L1-penalized linear regression in the interaction basis recover the secret parity subset?

## Hypothesis

If we expand input x in {-1,+1}^n to all C(n,k) interaction terms (products of k input bits), the parity function becomes a single nonzero coefficient in this basis. LASSO's sparsity-inducing L1 penalty should recover exactly that one coefficient, solving sparse parity without neural nets or gradient descent.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 50 |
| k_sparse | 3, 5 |
| method | sklearn Lasso (alpha=0.1) + LassoCV (5-fold) |
| n_samples | 500 (k=3, n=20), 1000 (k=3, n=50), 2000 (k=5) |
| seeds | 42, 43, 44 |
| max_iter | 10,000 |
| fit_intercept | False |

## Results

| Metric | Value |
|--------|-------|
| Accuracy (all configs) | 100% (9/9 runs) |
| Nonzero coefficients | Exactly 1 (all runs) |
| LASSO coef (alpha=0.1) | 0.9000 (bias from L1 shrinkage) |
| LassoCV coef (alpha=0.001) | 0.9990 (near-perfect) |
| Wall time n=20,k=3 | 0.28s (with CV), 0.011s (LASSO only) |
| Wall time n=50,k=3 | 11.7s (with CV), 0.33s (LASSO only) |
| Wall time n=20,k=5 | 17.3s (with CV), 0.48s (LASSO only) |
| Weighted ARD (n=20,k=3) | 861,076 |

## Key Table

| Config | Method | Acc | Time (LASSO only) | Time (with CV) | Features |
|--------|--------|-----|--------------------|----------------|----------|
| n=20,k=3 | LASSO | 3/3 | 0.005s | 0.28s | 1,140 |
| n=50,k=3 | LASSO | 3/3 | 0.15s | 11.7s | 19,600 |
| n=20,k=5 | LASSO | 3/3 | 0.21s | 17.3s | 15,504 |
| n=20,k=3 | Fourier | 3/3 | 0.009s | -- | 1,140 |
| n=50,k=3 | Fourier | 3/3 | 0.16s | -- | 19,600 |
| n=20,k=5 | Fourier | 3/3 | 0.14s | -- | 15,504 |
| n=20,k=3 | SGD | 100% | 0.12s | -- | 20 raw |
| n=50,k=3 | SGD direct | 54% | -- | -- | FAIL |
| n=20,k=3 | Random search | 5/5 | 0.011s | -- | -- |

## Alpha Sweep (n=20, k=3, seed=42)

| Alpha | Nonzero | Coef | Correct | Time |
|-------|---------|------|---------|------|
| 0.001 | 1 | 0.999 | Yes | 0.018s |
| 0.01 | 1 | 0.990 | Yes | 0.009s |
| 0.05 | 1 | 0.950 | Yes | 0.012s |
| 0.1 | 1 | 0.900 | Yes | 0.008s |
| 0.2 | 1 | 0.800 | Yes | 0.007s |
| 0.5 | 1 | 0.500 | Yes | 0.007s |
| 1.0 | 0 | 0.000 | No | 0.008s |

## Analysis

### What worked

- **Perfect accuracy on every config**: 9/9 runs across all three configs and seeds. LASSO finds exactly 1 nonzero coefficient corresponding to the true secret subset.
- **LASSO is robust to alpha**: any value from 0.001 to 0.5 works. Only alpha=1.0 fails (shrinks everything to zero). The problem is so well-separated that LASSO barely needs tuning.
- **LASSO only (no CV) is fast**: 0.005s for n=20/k=3 and 0.15s for n=50/k=3. Competitive with Fourier when you skip cross-validation.
- **Solves n=50,k=3 trivially** -- the same config where SGD fails outright (54%).
- **Mathematically elegant**: the parity function y = x_{i1} * x_{i2} * x_{i3} is literally a linear function in the interaction basis. LASSO is the right tool for sparse linear recovery.

### What didn't work

- **LassoCV dominates wall time**: 5-fold cross-validation is expensive. For n=50/k=3, LASSO itself takes 0.15s but LassoCV adds 12.7s. For n=20/k=5, LASSO takes 0.21s but CV adds 16.9s. CV is unnecessary here given LASSO's robustness to alpha.
- **Slower than Fourier**: Fourier solves n=20/k=3 in 0.009s vs LASSO's 0.005s (comparable), but Fourier handles n=50/k=3 in 0.16s vs LASSO's 0.15s + feature expansion. Once you add feature expansion overhead, they are similar for LASSO-only but Fourier wins on simplicity.
- **L1 shrinkage biases the coefficient**: at alpha=0.1 the recovered coefficient is 0.90 instead of 1.0. Not a problem for identification (largest coefficient wins), but the coefficient magnitude is systematically underestimated.
- **Feature expansion is O(n_samples * C(n,k))**: same combinatorial cost as Fourier. LASSO does not avoid the combinatorial blowup -- it just uses a different algorithm on the same feature matrix.
- **ARD is high** (861,076): the feature expansion creates a large matrix that must be read in full. Similar cache behavior to Fourier.

### Surprise

- **Exactly 1 nonzero coefficient every time**: LASSO's sparsity is perfect here. In typical LASSO applications you get some spurious nonzeros, but the interaction features for different subsets are nearly orthogonal (uncorrelated when inputs are uniform {-1,+1}), so the irrepresentable condition holds and LASSO achieves exact support recovery.
- **Alpha range is enormous**: anything from 0.001 to 0.5 works (500x range). The true coefficient has magnitude 1.0 and all spurious correlations are ~0, giving a massive signal-to-noise ratio. LASSO barely needs to try.
- **LassoCV always picks alpha=0.001**: the smallest alpha in the default grid, because there is essentially no noise to regularize against.

## Open Questions

- Can we skip the full feature expansion and instead use a screening rule? (e.g., compute marginal correlations first, then only expand promising subsets)
- How does LASSO scale to k=7 or k=10? C(20,7) = 77,520 and C(20,10) = 184,756 features -- the expansion itself becomes the bottleneck
- Elastic Net (L1 + L2) comparison: does adding L2 help or hurt when the true solution is maximally sparse?
- LASSO path (lars algorithm) could be faster than coordinate descent for this problem since there is exactly 1 nonzero

## Files

- Experiment: `src/sparse_parity/experiments/exp_lasso.py`
- Results: `results/exp_lasso/results.json`

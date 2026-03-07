# Experiment exp_random_proj: Random Projections + Correlation for Sparse Parity

**Date**: 2026-03-06
**Status**: SUCCESS
**Answers**: Monte Carlo Walsh-Hadamard with early stopping -- does random sampling beat exhaustive Fourier?

## Hypothesis

If we randomly sample k-subsets and compute Walsh-Hadamard correlation (corr = mean(y * prod(x[:, subset]))), we can find the secret with early stopping (|corr| > 0.9) using far fewer than C(n,k) evaluations. Expected ~C(n,k)/2 on average (geometric distribution), but with high variance that sometimes yields large speedups.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 50, 100, 200 |
| k_sparse | 3, 5 |
| method | random k-subset sampling + Walsh-Hadamard correlation |
| corr_threshold | 0.9 (early stopping) |
| n_samples | 500 |
| seeds | 42, 43, 44, 45, 46 (5 seeds) |
| max_tries | 50k-5M depending on config |

## Results

| Config | C(n,k) | RP avg tries | RP % of C(n,k) | RP avg time | Fourier time | Speedup |
|--------|--------|-------------|----------------|-------------|--------------|---------|
| n=20, k=3 | 1,140 | 565 | 49.6% | 0.013s | 0.010s | 0.8x |
| n=50, k=3 | 19,600 | 6,342 | 32.4% | 0.113s | 0.166s | 1.5x |
| n=20, k=5 | 15,504 | 6,238 | 40.2% | 0.134s | 0.142s | 1.1x |
| n=100, k=3 | 161,700 | 99,908 | 61.8% | 2.09s | N/A | -- |
| n=200, k=3 | 1,313,400 | 815,849 | 62.1% | 19.8s | N/A | -- |

## Variance Analysis (5 seeds)

| Config | Tries mean | Tries std | Tries min | Tries max | Time std |
|--------|-----------|-----------|-----------|-----------|----------|
| n=20, k=3 | 565 | 350 | 62 | 1,029 | 0.008s |
| n=50, k=3 | 6,342 | 3,654 | 52 | 11,035 | 0.069s |
| n=20, k=5 | 6,238 | 5,417 | 427 | 13,382 | 0.127s |
| n=100, k=3 | 99,908 | 26,933 | 47,525 | 121,556 | 0.693s |
| n=200, k=3 | 815,849 | 284,812 | 383,591 | 1,184,819 | 9.20s |

## Analysis

### What worked

- **100% solve rate across all configs and all seeds** (25/25 runs solved). The correlation test is perfectly reliable -- true subset always has corr=1.0, all others are near zero.
- **Saves ~40-70% of subset evaluations vs exhaustive Fourier** on average. The mean fraction of C(n,k) tested ranges from 32.4% (n=50/k=3) to 62.1% (n=200/k=3).
- **Best case can be dramatically faster**: n=50/k=3 with seed=45 found the secret in just 52 tries (0.3% of C(n,k)). n=20/k=5 with seed=42 found it in 427 tries (2.8% of C(n,k)).
- **Scales to n=200/k=3** (1.3M possible subsets) -- solves in ~20s average, testing only ~62% of the search space.
- **n=50/k=3 solved trivially** -- the config where SGD fails (54%) without curriculum learning.

### What didn't work

- **Modest average speedup over exhaustive Fourier**: only 1.1-1.5x on configs where both were measured. On n=20/k=3, Fourier is actually faster (0.8x) because the overhead of random sampling + duplicate checking outweighs the savings from early stopping when C(n,k) is small.
- **High variance**: standard deviation of tries is 40-87% of the mean. Some seeds get lucky (2.8% of C(n,k)), others unlucky (90.3%). This is inherent to the geometric distribution.
- **ARD is terrible** (same as Fourier): weighted ARD of 670K for n=20/k=3. Each subset evaluation reads the full dataset, so there is no temporal locality.
- **Duplicate checking overhead**: with large search spaces, the `tried` set grows large and random collisions increase, though this was not a major bottleneck in practice.

### Surprise

- **The speedup is underwhelming for small C(n,k)**. When C(n,k) = 1,140 (n=20/k=3), the exhaustive Fourier is actually faster because it uses `itertools.combinations` (optimized C) vs random sampling with duplicate detection. The random projection approach only wins when C(n,k) is large enough that 30-60% savings matter.
- **Variance is the real story**: the best-case n=50/k=3 run (seed=45) found the secret in 52 tries = 0.3% of C(n,k), finishing in 0.8ms. The worst case (seed=46) took 11,035 tries = 56.3%. If you're lucky, random projections are 100x faster than Fourier. If you're unlucky, they're barely faster.
- **n=200/k=3 is tractable** for random projections (20s) but would take exhaustive Fourier longer. At this scale the ~40% savings matter.

## Comparison with Other Methods

| Config | Random Proj | Fourier (exhaustive) | Random Search (exp_evolutionary) | SGD |
|--------|------------|---------------------|----------------------------------|-----|
| n=20, k=3 | 565 tries / 0.013s | 1,140 subsets / 0.010s | 881 tries / 0.011s | ~5 epochs / 0.12s |
| n=50, k=3 | 6,342 tries / 0.113s | 19,600 subsets / 0.166s | 11,291 tries / 0.142s | FAIL (54%) |
| n=20, k=5 | 6,238 tries / 0.134s | 15,504 subsets / 0.142s | 18,240 tries / 0.426s | 14 epochs (n=5000) |

Random projections tests fewer subsets than both exhaustive Fourier and the random search from exp_evolutionary. The difference from exp_evolutionary's random search is that this approach uses correlation (a statistical test) rather than exact match, plus it does duplicate avoidance.

## Open Questions

- Can we combine random projections with feature pre-screening (e.g., check individual bit correlations first, narrow candidates, then only sample from top bits)?
- What is the crossover point where random projections reliably beat exhaustive Fourier in wall time? Appears to be around C(n,k) > 10,000.
- Could we use adaptive sampling (e.g., if a subset has moderate correlation, mutate it) to get the best of both random projections and evolutionary search?

## Files

- Experiment: `src/sparse_parity/experiments/exp_random_proj.py`
- Results: `results/exp_random_proj/results.json`

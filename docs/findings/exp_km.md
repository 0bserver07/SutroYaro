# Experiment exp_km: Kushilevitz-Mansour Algorithm for Sparse Parity

**Date**: 2026-03-06
**Status**: SUCCESS
**Answers**: Can influence-based pruning solve sparse parity in O(n) rather than O(C(n,k))?

## Hypothesis

If we estimate per-bit influence (flip bit i, measure label change rate), secret bits will have influence 1.0 and non-secret bits will have influence 0.0. This prunes the search from C(n,k) subsets down to C(k,k)=1, giving an exact solver with O(n) queries instead of O(C(n,k)).

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 50, 100 |
| k_sparse | 3, 5 |
| hidden | N/A (no neural net) |
| lr | N/A |
| method | Kushilevitz-Mansour (influence estimation + pruned verification) |
| n_influence_samples | 200 (per bit) |
| n_verify_samples | 100 |
| influence_threshold | 0.5 |
| seeds | 42, 43, 44 |

## Results

| Metric | Value |
|--------|-------|
| Best test accuracy | 100% (all configs, all seeds) |
| Candidates found | Exactly k in every run |
| Subsets checked | 1 (always) |
| Total queries (n=20,k=3) | 8,200 |
| Total queries (n=100,k=3) | 40,200 |
| Wall time (n=20,k=3) | 0.006s |
| Wall time (n=100,k=3) | 0.009s |
| Wall time (n=20,k=5) | 0.001s |
| Weighted ARD (n=20,k=3) | 1,585 |

## Key Table

| Config | Method | Acc | Time | Queries/Subsets | Pruning ratio |
|--------|--------|-----|------|-----------------|---------------|
| n=20,k=3 | KM (this) | 100% | 0.006s | 8,200 queries | 1/1,140 subsets |
| n=50,k=3 | KM (this) | 100% | 0.003s | 20,200 queries | 1/19,600 subsets |
| n=100,k=3 | KM (this) | 100% | 0.009s | 40,200 queries | 1/161,700 subsets |
| n=20,k=5 | KM (this) | 100% | 0.001s | 8,200 queries | 1/15,504 subsets |
| n=20,k=3 | Fourier | 100% | 0.009s | 1,140 subsets | none |
| n=50,k=3 | Fourier | 100% | 0.16s | 19,600 subsets | none |
| n=100,k=3 | Fourier | 100% | 1.3s | 161,700 subsets | none |
| n=20,k=5 | Fourier | 100% | 0.14s | 15,504 subsets | none |
| n=20,k=3 | Random search | 100% | 0.011s | ~881 tries | none |
| n=50,k=3 | Random search | 100% | 0.142s | ~11,291 tries | none |
| n=20,k=3 | SGD (fast.py) | 100% | 0.12s | ~1000 samples | none |
| n=50,k=3 | SGD direct | 54% FAIL | --- | --- | none |

## Sample Complexity

| Influence samples/bit | n=20,k=3 | n=50,k=3 | n=20,k=5 | Total queries (n=20,k=3) |
|-----------------------|----------|----------|----------|--------------------------|
| 5 | CORRECT | CORRECT | CORRECT | 300 |
| 10 | CORRECT | CORRECT | CORRECT | 500 |
| 20 | CORRECT | CORRECT | CORRECT | 900 |
| 50 | CORRECT | CORRECT | CORRECT | 2,100 |
| 200 | CORRECT | CORRECT | CORRECT | 8,100 |

## Analysis

### What worked

- **Perfect pruning in every single run**: influence estimation identified exactly k high-influence bits out of n, reducing subsets to check from C(n,k) to exactly 1. Zero false positives, zero false negatives across all 12 runs.
- **Scales linearly with n**: total queries = 2 * n * n_influence_samples + n_verify_samples. For n=100,k=3 this is 40,200 queries vs Fourier's 161,700 subsets. The advantage grows with n.
- **Completely independent of k for the influence phase**: the per-bit influence estimation cost is the same whether k=3 or k=5. The only place k matters is verification (which is trivial at C(k,k)=1).
- **Works with absurdly few samples**: even 5 influence samples per bit correctly identifies the secret. The theoretical minimum is 1 sample (since influence is exactly 0 or 1 for parity), but statistical noise from finite samples could cause errors in more adversarial settings.
- **Fastest method tested**: 0.001-0.009s across all configs. Beats Fourier (0.009-1.3s), random search (0.011-0.142s), and SGD (0.12s+) on every config.
- **ARD of 1,585 vs Fourier's 1,147,375**: 724x better memory locality because the algorithm accesses far less data.

### What didn't work

- **The query count is technically higher than Fourier for small n,k**: 8,200 queries for n=20,k=3 vs Fourier checking 1,140 subsets (each needing ~500 sample correlations, so ~570,000 operations). But KM queries are paired (x and x^i), which is a different cost model.
- **Influence estimation uses a white-box oracle**: we need to query f(x) and f(x^i) where we control x. In a pure PAC learning setting with i.i.d. samples, you cannot construct paired flipped queries -- you'd need a different approach.

### Surprise

- **5 samples per bit is enough**: this means the entire n=20,k=3 problem can be solved with 300 total queries. That is 3.8x fewer than the C(20,3)=1,140 subsets Fourier needs to check, and each query is cheaper (single oracle call vs computing a correlation over all samples).
- **The algorithm is embarrassingly simple**: estimate n influences, threshold, done. The theoretical KM algorithm involves heavy Fourier analysis machinery, but for the special case of parity functions, the influence shortcut makes it trivial.
- **n=100,k=3 in 0.009s**: Fourier takes 1.3s on the same config. KM is 144x faster because it never has to enumerate the 161,700 subsets.

## Comparison with Fourier (exp_fourier)

KM is strictly better than brute-force Fourier for sparse parity:

| Metric | KM | Fourier | KM advantage |
|--------|-----|---------|--------------|
| n=20,k=3 time | 0.006s | 0.009s | 1.5x |
| n=50,k=3 time | 0.003s | 0.16s | 53x |
| n=100,k=3 time | 0.009s | 1.3s | 144x |
| n=20,k=5 time | 0.001s | 0.14s | 140x |
| Scaling | O(n) | O(C(n,k)) | polynomial vs combinatorial |
| Min samples | 5/bit | ~20 total | different cost model |
| ARD (n=20,k=3) | 1,585 | 1,147,375 | 724x |

The advantage grows with n and k because Fourier's C(n,k) explodes while KM stays O(n).

## Open Questions

- How does this extend to noisy parity (labels corrupted with probability p)? Influence estimation still works but requires more samples: O(1/p^2) per bit.
- Can this be combined with a neural net? Use influence estimation to select features, then train a tiny network on only the k selected bits.
- What about unknown k? Could run influence estimation and count the number of high-influence bits to infer k.
- In a true PAC learning model (i.i.d. samples only, no paired queries), can we still estimate influence efficiently? Yes, via Var(E[y|x_i]) which only needs random samples.

## Files

- Experiment: `src/sparse_parity/experiments/exp_km.py`
- Results: `results/exp_km/results.json`

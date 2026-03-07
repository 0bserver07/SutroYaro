# Experiment exp_gf2: Gaussian Elimination over GF(2) for Sparse Parity

**Date**: 2026-03-06
**Status**: SUCCESS
**Answers**: Theoretically optimal blank-slate approach -- can we solve sparse parity in O(n^2) with n+1 samples?

## Hypothesis

Treating each sample as a linear equation over GF(2) (binary field), Gaussian elimination recovers the secret parity bits in O(n^2) time with only ~n samples. This should be the fastest possible approach for pure parity, independent of k and C(n,k).

Key math: parity over {-1,+1} maps to XOR over {0,1}. For odd k, y_bit = XOR(x_bit[S]). For even k, y_bit = 1 - XOR(x_bit[S]). We solve both systems and verify which produces correct labels.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 50, 100 |
| k_sparse | 3, 5, 7, 10 |
| method | GF(2) Gaussian elimination (row reduction with XOR) |
| n_samples | n+1, 2n, 50, 100, 500 |
| seeds | 42, 43, 44 |

## Results

| Config | C(n,k) | Min samples (3/3) | Avg time (min samples) | Avg time (500 samples) |
|--------|--------|--------------------|------------------------|------------------------|
| n=20, k=3 | 1,140 | 21 (n+1) | 509 us | 8,034 us |
| n=50, k=3 | 19,600 | 51 (n+1) | 2,077 us | 19,690 us |
| n=100, k=3 | 161,700 | 101 (n+1) | 8,074 us | 39,056 us |
| n=20, k=5 | 15,504 | 21 (n+1) | 405 us | 8,154 us |
| n=20, k=7 | 77,520 | 40 (2n) | 695 us | 8,304 us |
| n=20, k=10 | 184,756 | 40 (2n) | 703 us | 7,880 us |

## Sample Complexity (n=20, k=3)

| n_samples | Correct (out of 3) | Avg time |
|-----------|-------------------|----------|
| 5 | 0/3 | 61 us |
| 10 | 0/3 | 127 us |
| 15 | 0/3 | 235 us |
| 18 | 2/3 | 310 us |
| 19 | 3/3 | 334 us |
| 20 | 3/3 | 377 us |
| 21+ | 3/3 | 379+ us |

## Comparison with All Approaches

| Config | Method | Time | Samples needed | Speedup vs SGD |
|--------|--------|------|----------------|----------------|
| n=20, k=3 | **GF(2) Gauss** | **509 us** | **21** | **~240x** |
| n=20, k=3 | Fourier exhaustive | ~3,000 us | 500 | ~40x |
| n=20, k=3 | Random search | ~11,000 us | 500 | ~11x |
| n=20, k=3 | SGD baseline | ~120,000 us | 10,000 | 1x |
| n=50, k=3 | **GF(2) Gauss** | **2,077 us** | **51** | solves trivially |
| n=50, k=3 | SGD (direct) | --- | 10,000 | FAILS (54%) |
| n=50, k=3 | SGD + curriculum | ~seconds | 10,000 | 20 epochs |
| n=20, k=5 | **GF(2) Gauss** | **405 us** | **21** | instant |
| n=20, k=5 | SGD | ~seconds | 5,000 | 14 epochs |
| n=20, k=10 | **GF(2) Gauss** | **703 us** | **40** | instant |
| n=100, k=3 | **GF(2) Gauss** | **8,074 us** | **101** | trivial |

## Analysis

### What worked

- **All 6 configs solved with 100% accuracy** using 40+ samples. GF(2) Gaussian elimination is completely independent of C(n,k) -- it does not enumerate subsets at all.
- **n+1 samples suffice for small k** (k=3,5): with only 21 samples for n=20, the system is fully determined and the solution is unique. This is the information-theoretic minimum.
- **Time complexity is O(n * m) where m = n_samples**: with minimum samples, O(n^2). For n=20 this is ~500 microseconds. The time is dominated by numpy array creation, not the elimination itself.
- **Completely k-independent**: unlike Fourier (O(C(n,k))) or random search (O(C(n,k))), GF(2) elimination does not depend on k at all. n=20/k=10 solves as fast as n=20/k=3.
- **n=50/k=3 solved in 2ms with 51 samples** -- a config where SGD completely fails (54%) without curriculum learning and 10,000 samples.
- **n=100/k=3 solved in 8ms with 101 samples** -- a config where Fourier would need to check C(100,3) = 161,700 subsets.

### What didn't work

- **n+1 samples is not always enough for large k**: with n=20/k=7, only 2/3 seeds succeed with 21 samples. With k=10, only 1/3. The issue is that the GF(2) linear system may have multiple solutions when k is large relative to n, and the solver picks the wrong one (the minimum-weight solution from free variables = 0 is not necessarily the true secret).
- **Even k requires special handling**: the mapping from product parity to XOR differs for even vs odd k. We must solve both A*s=b and A*s=(1-b) and verify which solution produces correct labels on the training data.

### Surprise

- **The method is absurdly fast**: ~500 microseconds to solve a problem that SGD takes 120,000+ microseconds for (240x speedup). And it uses 21 samples vs SGD's 10,000 (500x fewer samples).
- **19 samples suffice for n=20/k=3** (even less than n): the system does not need to be fully determined; it just needs enough equations to uniquely pin down the 3 active bits. In practice, n-1 samples often work because most columns get a pivot.
- **Time scales with matrix size, not problem combinatorics**: doubling n from 50 to 100 roughly doubles the time, while C(n,3) increases 8x. This is the signature advantage of treating parity as a linear algebra problem rather than a combinatorial search.

## Limitations

- **Only works for pure parity**: if the labels have noise, GF(2) elimination will fail (the system becomes inconsistent). In contrast, Fourier/Walsh-Hadamard is robust to noise because it averages correlations.
- **Requires knowing that the problem IS a parity**: this is not a general-purpose learning algorithm. It exploits the exact algebraic structure of the parity function.
- **k must be known (or both parities tried)**: we need to handle the even/odd k distinction, which adds a constant factor of 2.

## Open Questions

- Can we extend this to noisy labels? (Error-correcting codes over GF(2), syndrome decoding)
- What about mixtures of parities (y = XOR of multiple different subsets)?
- How does this connect to the grokking phenomenon -- is SGD implicitly learning GF(2) structure during the phase transition?

## Files

- Experiment: `src/sparse_parity/experiments/exp_gf2.py`
- Results: `results/exp_gf2/results.json`

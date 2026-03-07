# Experiment exp_mdl: Minimum Description Length for Sparse Parity

**Date**: 2026-03-06
**Status**: SUCCESS
**Approach**: #17 -- MDL / Compression. The best compressor of the label sequence is the one that knows the secret bits.

## Hypothesis

For each candidate k-subset S, compute the parity under S and measure how well it compresses the label sequence. The true subset produces a perfectly compressible (zero-entropy) residual; wrong subsets produce ~50% error rate residuals that cost ~n_samples bits. MDL is more general than Fourier correlation: it works for any deterministic labeling function and degrades gracefully under label noise.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 50 |
| k_sparse | 3, 5 |
| n_samples | 500 |
| seeds | 42, 43, 44 |
| noise_rate | 0% (clean), 5% (robustness test) |
| DL formula | n_samples * H(residual_rate), H = binary entropy |
| Comparison | Fourier/Walsh-Hadamard correlation on same data |

## Results

### Clean data (no noise)

| Config | C(n,k) | MDL correct | MDL avg time | Fourier correct | Fourier avg time |
|--------|--------|-------------|--------------|-----------------|------------------|
| n=20, k=3 | 1,140 | 3/3 | 0.0133s | 3/3 | 0.0118s |
| n=50, k=3 | 19,600 | 3/3 | 0.2673s | 3/3 | 0.2090s |
| n=20, k=5 | 15,504 | 3/3 | 0.2542s | 3/3 | 0.1868s |

- True subset DL = 0.0 bits (perfect compression) in all cases
- Wrong subset DL averages ~499.3 bits (out of 500 samples), confirming near-chance residual rate

### Noise robustness (5% label flips)

| Config | MDL correct | MDL best DL (bits) | Fourier correct | Fourier best corr |
|--------|-------------|--------------------|-----------------|--------------------|
| n=20, k=3 | 3/3 | 157.0 avg | 3/3 | 0.887 avg |
| n=50, k=3 | 3/3 | 107.0 avg | 3/3 | 0.932 avg |
| n=20, k=5 | 3/3 | 157.0 avg | 3/3 | 0.887 avg |

- Under 5% noise, the true subset DL rises from 0 to ~107-164 bits (= 500 * H(0.05) ~ 143 bits) but remains far below wrong subsets (~499 bits)
- Both MDL and Fourier find the correct subset under 5% noise -- separation is large enough

### MemTracker (n=20, k=3, seed=42)

| Metric | Value |
|--------|-------|
| Total floats accessed | 2,290,500 |
| Reads | 2,280 |
| Writes | 2 |
| Weighted ARD | 1,147,375 floats |

ARD is large because every subset scan re-reads x and y from scratch (streaming pattern, one pass per subset).

## Analysis

### What worked

- **MDL finds the exact secret subset on all configs with 100% accuracy**, matching Fourier
- **Noise robustness confirmed**: 5% label noise does not degrade MDL -- the gap between true subset DL (~150 bits) and wrong subset DL (~499 bits) is enormous
- **Zero-entropy signal is unambiguous**: the true subset compresses to exactly 0 bits on clean data, while even the next-best wrong subset sits at ~490+ bits
- **MDL is conceptually more general**: unlike Fourier correlation which exploits the multiplicative structure of parity, MDL only requires that the true hypothesis produces a low-entropy residual -- it would work for any deterministic labeling function

### What didn't work

- **MDL is ~30% slower than Fourier** on the same data: MDL computes residual_rate + binary_entropy per subset, while Fourier computes a single dot product + abs. The entropy computation (with log2) is more expensive than a mean
- **No practical advantage over Fourier for parity specifically**: both achieve 100% accuracy on all configs. MDL's generality does not help when the labeling function is known to be parity

### Key insight

MDL and Fourier are equivalent for sparse parity in the noiseless case: both produce a binary decision (true subset scores 0/1.0, wrong subsets score ~499/~0). Under noise, both degrade gracefully. The MDL framework is theoretically more principled (it directly measures compression quality rather than correlation), but for parity the distinction is academic.

### Quantitative comparison with other approaches

| Method | n=20/k=3 | n=50/k=3 | n=20/k=5 |
|--------|----------|----------|----------|
| MDL | 0.013s, 100% | 0.267s, 100% | 0.254s, 100% |
| Fourier | 0.012s, 100% | 0.209s, 100% | 0.187s, 100% |
| Random search | 0.011s, 100% | 0.142s, 100% | 0.426s, 100% |
| SGD (baseline) | 0.12s, 100% | FAIL (54%) | 14 epochs |

## Open Questions

- Would MDL outperform Fourier on a non-parity labeling function (e.g., threshold, majority, or XOR-of-subsets)?
- Can we speed up MDL by computing residual rates in batch (vectorized over subsets)?
- At what noise level does MDL start failing? Theoretically when 500 * H(noise_rate) approaches 500 * H(0.5), i.e. noise > ~40%
- Could a two-stage approach work: first screen by MDL, then verify by Fourier?

## Files

- Experiment: `src/sparse_parity/experiments/exp_mdl.py`
- Results: `results/exp_mdl/results.json`

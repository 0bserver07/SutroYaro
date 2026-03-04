# Experiment exp_fourier: Fourier/Walsh-Hadamard Solver for Sparse Parity

**Date**: 2026-03-04
**Status**: SUCCESS
**Answers**: Blank-slate approach — can we solve sparse parity without neural nets?

## Hypothesis

If we compute Walsh-Hadamard correlations for every k-subset, the true secret will have correlation ~1.0 while all others are ~0, giving an exact solver in O(C(n,k) * n_samples) time with no training.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 50, 100, 200 |
| k_sparse | 3, 5, 7 |
| hidden | N/A (no neural net) |
| lr | N/A |
| wd | N/A |
| batch_size | N/A |
| max_epochs | N/A |
| n_train | 500 (20 sufficient for k=3) |
| seed | 42, 43, 44 |
| method | fourier (Walsh-Hadamard exhaustive search) |

## Results

| Metric | Value |
|--------|-------|
| Best test accuracy | 100% (all configs) |
| Epochs to >90% | N/A (single pass) |
| Wall time (n=20,k=3) | 0.009s |
| Wall time (n=50,k=3) | 0.16s |
| Wall time (n=20,k=5) | 0.14s |
| Weighted ARD (n=20,k=3) | 1,147,375 |
| ARD improvement vs baseline | N/A (different algorithm class) |

## Key Table

| Config | Method | Acc | Time | Subsets | Samples |
|--------|--------|-----|------|---------|---------|
| n=20,k=3 | Fourier | 100% | 0.009s | 1,140 | 500 |
| n=20,k=3 | SGD (fast.py) | 100% | 0.12s | — | 1000 |
| n=50,k=3 | Fourier | 100% | 0.16s | 19,600 | 500 |
| n=50,k=3 | SGD direct | 54% FAIL | — | — | 200 |
| n=50,k=3 | Curriculum | >90% | — | — | — |
| n=20,k=5 | Fourier | 100% | 0.14s | 15,504 | 500 |
| n=20,k=5 | SGD (n=5000) | 100% | — | — | 5000 |
| n=100,k=3 | Fourier | 100% | 1.3s | 161,700 | 500 |
| n=200,k=3 | Fourier | 100% | 10.8s | 1,313,400 | 500 |
| n=20,k=7 | Fourier | 100% | 0.7s | 77,520 | 500 |

## Analysis

### What worked

- Perfect accuracy on every config tested. Correlation for the true subset is always exactly 1.0.
- **13x faster than SGD on n=20,k=3** (0.009s vs 0.12s).
- **Solves n=50,k=3 trivially** where SGD fails outright (54%) and curriculum needs 20 epochs.
- **Solves n=20,k=7** which SGD has never been tested on — 77,520 subsets in 0.7s.
- Only needs ~20 samples for k=3 (vs 500-5000 for SGD). Sample complexity is O(1/epsilon^2), independent of n.
- Scales to n=200,k=3 in 10.8s (1.3M subsets).

### What didn't work

- **ARD is terrible**: 1,147,375 weighted ARD vs 17,976 for SGD. The algorithm reads the entire dataset for each subset — pure streaming, no locality.
- **Combinatorial wall at high k**: C(n,k) grows fast. C(50,5) = 2.1M, C(100,5) = 75M, C(50,7) = 99M. The method is fundamentally O(C(n,k)) which is polynomial in n for fixed k but explodes with k.
- Not a learning algorithm — it's a brute-force search. Doesn't generalize or transfer.

### Surprise

- The Fourier solver is **13x faster than the best SGD** on n=20,k=3. We spent weeks optimizing SGD and a trivial brute-force beats it. The neural net was solving an easy problem the hard way.
- Only **20 samples** are needed for reliable k=3 detection. SGD uses 500-5000.

## Open Questions (for next experiment)

- Can we prune the subset search with single-variable correlations first? (Check each bit's individual correlation, discard low ones, then only check k-subsets of the top candidates.)
- Hybrid: use Fourier to find the secret, then a trivial classifier (just compute the product) — what's the total energy cost vs SGD?
- At what (n,k) does SGD actually beat Fourier in wall time?

## Files

- Experiment: `src/sparse_parity/experiments/exp_fourier.py`
- Results: `results/exp_fourier/results.json`

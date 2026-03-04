# Experiment sign_sgd: Sign SGD for k=5 Sparse Parity

**Date**: 2026-03-04
**Status**: SUCCESS (with caveats)
**Answers**: Open Question #1 — "Can Sign SGD solve k=5?"

## Hypothesis

If we replace gradient descent with sign(gradient) descent, then k=5 sparse parity will become solvable because Sign SGD needs n^{k-1} samples instead of n^k (Kou et al. 2024).

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20 (also 30) |
| k_sparse | 3 and 5 |
| hidden | 200 (k=3), 500 (k=5) |
| lr | 0.01 (sign), 0.1 (standard) |
| wd | 0.01 |
| batch_size | 32 |
| max_epochs | 200-500 |
| n_train | 1000-50000 |
| seed | 42, 43, 44 |
| method | sign-sgd vs standard |

## Results

| Metric | Value |
|--------|-------|
| Best test accuracy (sign, k=5, 5K) | 100% |
| Epochs to >90% (sign, k=5, 5K) | 7 (vs 14 standard) |
| Best test accuracy (std, k=5, 5K) | 100% |
| Wall time (sign, k=5, 5K) | 0.44s |
| Wall time (std, k=5, 5K) | 0.81s |

## Key Table

| Config | Method | n_train | Best Acc | Ep->90% | Time(s) | Solved |
|--------|--------|---------|----------|---------|---------|--------|
| n=20,k=3 | standard | 1000 | 100% | 34 | 0.68 | 3/3 |
| n=20,k=3 | sign (lr=0.01) | 1000 | 99.7% | 26 | 0.42 | 3/3 |
| n=20,k=5 | standard | 5000 | 100% | 14 | 0.81 | 3/3 |
| n=20,k=5 | sign (lr=0.01) | 5000 | 100% | 7 | 0.44 | 3/3 |
| n=20,k=5 | sign (lr=0.01) | 20000 | 100% | 2 | 0.81 | 3/3 |
| n=20,k=5 | sign (lr=0.001) | 5000 | 98.1% | 145 | 14.06 | 3/3 |
| n=30,k=3 | standard | 2000 | 100% | 22 | 0.18 | 3/3 |
| n=30,k=3 | sign (lr=0.01) | 2000 | 100% | 15 | 0.36 | 3/3 |

## Analysis

### What worked

- Sign SGD solves k=5 reliably (100% across 3 seeds)
- Sign SGD converges 2x faster to 90% on k=5 (7 vs 14 epochs with n_train=5000)
- With 20K samples, Sign SGD reaches 90% in just 2 epochs
- For k=3, Sign SGD reaches 90% slightly faster (26 vs 34 epochs on n=20; 15 vs 22 on n=30)

### What didn't work

- Sign SGD with lr=0.001 is very slow (145 epochs to 90% on k=5 with 5K samples)
- Sign SGD oscillates near the top on k=3 — reaches 99% but struggles to hit 100% (due to fixed step size)
- lr=0.001 on k=3 fails entirely (78% max in 200 epochs)

### Surprise

**Standard SGD ALSO solves k=5 with enough data.** The earlier exp_d finding that "k=5 is categorically impractical" was wrong — it used n_train=200 (= 10*n_bits). With n_train=5000, standard SGD solves n=20/k=5 at 100% in 14 epochs. The bottleneck was training data, not the optimization algorithm.

This means the n^k sample complexity bound is pessimistic in practice. With n_train=5000 << n^k=3,200,000, standard SGD still solves it.

Sign SGD's advantage is convergence speed (2x fewer epochs to 90%) and data efficiency (reaches 90% in 2 epochs with 20K samples vs much more with standard SGD at lower lr).

## Open Questions (for next experiment)

- What is the actual minimum n_train for k=5 with standard SGD vs sign SGD? Binary search to find the threshold.
- Does Sign SGD help at k=7 or k=9 where the gap should be larger?
- Can learning rate warmup or decay fix Sign SGD's oscillation problem near 100%?
- What does Sign SGD's ARD look like? The sign() operation itself is cheap, but does the convergence speed offset help energy?

## Files

- Experiment: `src/sparse_parity/experiments/exp_sign_sgd.py`
- Results: `results/exp_sign_sgd/results.json`

# Experiment exp_evolutionary: Evolutionary/Random Search for Sparse Parity

**Date**: 2026-03-04
**Status**: SUCCESS
**Answers**: Blank slate approach — can sparse parity be solved without neural nets or gradients?

## Hypothesis

If we search directly over k-subsets of input bits, random search should need ~C(n,k) tries on average, and evolutionary search (with tournament selection + mutation) should beat that by exploiting partial fitness signal.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 50 |
| k_sparse | 3, 5 |
| method | random search, evolutionary search (pop=100-200, tournament=3) |
| n_train | 500 (k=3), 2000 (k=5) |
| seeds | 42, 43, 44, 45, 46 |

## Results

| Config | C(n,k) | Random: avg tries | Random: avg time | Evo: avg gens | Evo: avg time | Both solved |
|--------|--------|-------------------|------------------|---------------|---------------|-------------|
| n=20, k=3 | 1,140 | 881 | 0.011s | 18 | 0.041s | 5/5 |
| n=50, k=3 | 19,600 | 11,291 | 0.142s | 151 | 0.781s | 5/5 |
| n=20, k=5 | 15,504 | 18,240 | 0.426s | 74 | 0.552s | 5/5 |

## Key Table

| Method | n=20/k=3 | n=50/k=3 | n=20/k=5 |
|--------|----------|----------|----------|
| Random search | 881 tries / 0.011s | 11,291 tries / 0.142s | 18,240 tries / 0.426s |
| Evolutionary | 18 gens / 0.041s | 151 gens / 0.781s | 74 gens / 0.552s |
| SGD (baseline) | ~5 epochs / 0.12s | FAIL direct (54%) | 14 epochs (n_train=5000) |
| SGD + curriculum | — | 20 epochs | — |

## Analysis

### What worked

- **Both approaches solve all configs with 100% accuracy** — no approximate solutions, exact subset recovery
- **Random search is surprisingly competitive**: averages ~0.77x C(n,k) tries, which is expected (geometric distribution with p=1/C(n,k))
- **Random search is faster in wall time** than evolutionary for all tested configs because each trial is O(n_train*k) while evo must evaluate 100-200 candidates per generation
- **Evolutionary search uses far fewer fitness evaluations**: 18 vs 881 (n=20/k=3), 151 vs 11,291 (n=50/k=3), but each generation evaluates the full population
- **n=50/k=3 solved trivially** — this is the config that SGD fails on (54%) without curriculum learning

### What didn't work

- **Evolutionary search is slower in wall time** than random search despite fewer generations, because it evaluates a full population each generation (100-200 fitness calls per gen)
- **The fitness landscape is deceptive for evo**: a wrong k-subset gives ~50% fitness (random chance), so the signal for partial correctness is weak until you get 2 out of 3 bits right

### Surprise

- **Random search solves n=50/k=3 in 0.14s** — a config that SGD cannot solve directly (54% max in 200 epochs). The combinatorial approach completely bypasses the grokking/generalization problem because it checks exact parity match on training data.
- **n=20/k=5 solved by random search in 0.43s** — SGD needs 5000 training samples and 14 epochs. Random search only needs enough samples to verify uniqueness (~500 for k=3, 2000 for k=5).

## Comparison with SGD

The key insight: **SGD and random search solve fundamentally different problems**.
- SGD learns a *neural network* that computes parity — it must generalize from training data
- Random/evo search finds the *exact subset* — it just needs enough data to rule out false positives
- Random search complexity is O(C(n,k)) which is polynomial in n for fixed k
- SGD complexity depends on the grokking phase transition, which is harder to predict

For small k (3-5), random search is simpler, faster, and more reliable. SGD's advantage appears only when you need a differentiable model or when k is unknown.

## Open Questions (for next experiment)

- How does random search scale to k=7 or k=10? C(20,7) = 77,520 and C(20,10) = 184,756 — still feasible
- Can we use statistical tests (e.g., correlation between individual bits and y) to narrow the search space before enumeration?
- Hybrid approach: use random search to find candidate subsets, verify with a small neural net

## Files

- Experiment: `src/sparse_parity/experiments/exp_evolutionary.py`
- Results: `results/exp_evolutionary/results.json`

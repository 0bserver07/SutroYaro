# Experiment exp_genetic_prog: Genetic Programming for Sparse Parity

**Date**: 2026-03-06
**Status**: PARTIAL SUCCESS
**Answers**: Can GP evolve symbolic programs that compute sparse parity? Yes for small n, no for larger n/k.

## Hypothesis

GP can evolve expression trees of the form `sign(x[a] * x[b] * x[c])` to solve sparse parity. The discovered program would have zero learned parameters, zero memory footprint, and zero ARD. GP should find this via crossover and mutation over symbolic expression trees with primitives {multiply, negate, sign} and terminals {x[i], 1.0, -1.0}.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 50 |
| k_sparse | 3, 5 |
| pop_size | 200 |
| max_generations | 200 (n=20/k=3), 500 (others) |
| max_depth | 6-8 |
| tournament_size | 4 |
| crossover_rate | 0.7 |
| mutation_rate | 0.2 |
| point_mutation_rate | 0.1 |
| parsimony_coeff | 0.002 |
| n_train | 1000 (k=3), 2000 (k=5) |
| seeds | 42, 43, 44 |
| init method | ramped half-and-half |

## Results

| Config | C(n,k) | Solved | Avg gens | Avg time | Test acc | Params | ARD |
|--------|--------|--------|----------|----------|----------|--------|-----|
| n=20, k=3 | 1,140 | 3/3 | 75 | 0.98s | 100.0% | 0 | 0 |
| n=50, k=3 | 19,600 | 0/3 | --- | --- | 50.2% | 0 | 0 |
| n=20, k=5 | 15,504 | 0/3 | --- | --- | 49.8% | 0 | 0 |

### Example discovered programs (n=20, k=3)

| Seed | Secret | Found program | Correct vars | Depth |
|------|--------|--------------|--------------|-------|
| 42 | [0,15,17] | `mul(x[0], mul(x[15], x[17]))` | Yes | 3 |
| 43 | [6,9,19] | `mul(mul(x[19], mul(mul(x[18], x[18]), x[6])), x[9])` | Yes (x[18]^2=1) | 5 |
| 44 | [9,14,15] | `mul(sign(mul(x[9], x[14])), x[15])` | Yes | 4 |

## Analysis

### What worked

- **n=20/k=3 solved perfectly (3/3 seeds)** with 100% test accuracy. GP discovers the exact symbolic program.
- **Zero parameters**: the discovered programs are pure symbolic expressions with no learned weights.
- **Compact solutions**: seed 42 found `mul(x[0], mul(x[15], x[17]))` -- a depth-3 tree with 5 nodes, which is essentially the optimal solution.
- **GP finds algebraically equivalent solutions**: seed 43 found a program using x[18]^2 (which equals 1 for {-1,+1} inputs), effectively computing `x[6] * x[9] * x[19]`. GP exploits the algebraic structure of the domain.

### What didn't work

- **n=50/k=3 and n=20/k=5 completely failed** (0/3 seeds, ~50% accuracy = random chance).
- **The fitness landscape is a needle-in-a-haystack**: with {-1,+1} inputs, any wrong subset of variables gives exactly 50% accuracy. There is no gradient signal -- partial correctness (getting 2 out of 3 bits right) still gives ~50% accuracy because the wrong product is uncorrelated with the true parity.
- **GP converges prematurely**: diversity drops quickly (from ~170 to ~50 unique variable sets) but the population stagnates at ~55% accuracy. More generations do not help.
- **Larger search space is fatal**: with n=50, each variable appears in only ~6% of trees, making it very unlikely for crossover/mutation to assemble the right 3 variables without fitness guidance.

### Key insight: why GP fails where evolutionary subset search succeeds

The exp_evolutionary experiment solved n=50/k=3 easily (151 gens). The difference is fundamental:
- **Evolutionary subset search** represents candidates as k-subsets directly. Every candidate is a valid k-subset, and fitness evaluation is trivial (check if product matches labels).
- **GP** represents candidates as expression trees. The search space is vastly larger (all possible trees up to depth 6-8), and most of that space is irrelevant. GP must simultaneously discover the right structure (nested multiplications) AND the right variables.
- **The parity fitness landscape has zero gradient**: unlike problems where partial solutions provide partial fitness, sparse parity is an all-or-nothing function. This makes GP's crossover and mutation essentially random search through an enormous space.

### Comparison with other approaches

| Method | n=20/k=3 | n=50/k=3 | n=20/k=5 |
|--------|----------|----------|----------|
| GP (this exp) | 75 gens / 0.98s | FAIL | FAIL |
| Evo subset search | 18 gens / 0.04s | 151 gens / 0.78s | 74 gens / 0.55s |
| Random subset search | 881 tries / 0.01s | 11,291 / 0.14s | 18,240 / 0.43s |
| SGD (baseline) | ~5 epochs / 0.12s | FAIL direct | 14 epochs |

## Open Questions

- Would seeding the GP population with known good subtrees (e.g., `mul(x[i], x[j])` for all pairs) bootstrap the search?
- Could a hybrid approach use GP for structure search but evolutionary subset search for variable selection?
- Would strongly-typed GP that enforces "multiply only variables" reduce the search space enough?
- Is the needle-in-a-haystack problem inherent to GP on parity, or can fitness sharing / niching help?

## Files

- Experiment: `src/sparse_parity/experiments/exp_genetic_prog.py`
- Results: `results/exp_genetic_prog/results.json`

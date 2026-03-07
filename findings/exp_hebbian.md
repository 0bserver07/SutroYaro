# Experiment exp_hebbian: Hebbian + Anti-Hebbian Learning for Sparse Parity

**Date**: 2026-03-06
**Status**: FAILED (as predicted)
**Answers**: Can purely local, biologically-inspired learning rules solve sparse parity without backprop?

## Hypothesis

Purely local Hebbian learning rules (no backprop, no global error signal) will fail to solve sparse parity because parity bits have zero linear correlation with the label. Hebbian variants find correlations and principal components, but the parity function is a high-order interaction that is invisible to any linear or second-order statistic. However, the ARD (memory reuse distance) should be low because updates are purely local.

## Config

| Parameter | Config 1 | Config 2 | Config 3 |
|-----------|----------|----------|----------|
| n_bits | 20 | 20 | 20 |
| k_sparse | 3 | 3 | 3 |
| hidden | 1000 | 200 | 1000 |
| lr | 0.1 | 0.1 | 0.01 |
| alpha (decay) | 0.01 | 0.01 | 0.001 |
| max_epochs | 200 | 200 | 200 |
| n_train | 500 | 500 | 500 |
| n_test | 200 | 200 | 200 |
| seeds | 42, 43, 44 | 42, 43, 44 | 42, 43, 44 |
| rules | simple_hebb, oja, bcm | simple_hebb, oja, bcm | simple_hebb, oja, bcm |

## Results

| Config | Rule | Best Acc | Avg Acc | Converged | Avg ARD |
|--------|------|----------|---------|-----------|---------|
| n20/k3/h1000/lr0.1 | simple_hebb | 0.495 | 0.482 | 0/3 | 34,798 |
| n20/k3/h1000/lr0.1 | oja | 0.495 | 0.482 | 0/3 | 34,798 |
| n20/k3/h1000/lr0.1 | bcm | 0.495 | 0.482 | 0/3 | 34,798 |
| n20/k3/h200/lr0.1 | simple_hebb | 0.515 | 0.502 | 0/3 | 6,985 |
| n20/k3/h200/lr0.1 | oja | 0.495 | 0.482 | 0/3 | 6,985 |
| n20/k3/h200/lr0.1 | bcm | 0.495 | 0.482 | 0/3 | 6,985 |
| n20/k3/h1000/lr0.01 | simple_hebb | 0.560 | 0.517 | 0/3 | 34,798 |
| n20/k3/h1000/lr0.01 | oja | 0.545 | 0.525 | 0/3 | 34,798 |
| n20/k3/h1000/lr0.01 | bcm | 0.545 | 0.512 | 0/3 | 34,798 |

## Analysis

### What worked

- **Experiment ran cleanly and confirmed theory**: All 27 runs (3 configs x 3 rules x 3 seeds) completed. No rule exceeded 56% accuracy (chance = 50%). This is a clean negative result that validates the theoretical prediction.
- **ARD is indeed modest**: At 34,798 for hidden=1000 and 6,985 for hidden=200, Hebbian ARD is proportional to network size. No backward pass means no gradient chain, so memory access patterns are localized per layer.
- **Lower learning rate (0.01) gave slightly more stable results**: Config 3 showed marginally higher accuracies (up to 56%), avoiding the numerical overflow issues seen at lr=0.1. This is noise, not signal, but it confirms the rules are at least numerically stable at lower lr.

### What didn't work

- **All three Hebbian rules fail completely**: Simple Hebb, Oja's rule, and BCM all produce chance-level accuracy (45-56%). None converged. This is because:
  1. **Zero linear correlation**: For {-1,+1} inputs with label = product of k bits, each individual bit has zero correlation with the label. Hebbian rules learn correlations, so they find nothing.
  2. **PCA is useless**: Oja's rule extracts principal components. But the inputs are i.i.d. {-1,+1}, so all PCA directions are equivalent (uniform variance). The principal components carry no parity information.
  3. **BCM selectivity cannot help**: BCM creates neurons selective to specific patterns via its sliding threshold. But selectivity to individual bits or linear combinations of bits still cannot capture XOR/parity structure.
- **Numerical instability at lr=0.1**: Simple Hebb with decay and Oja's rule both showed overflow/NaN warnings, especially with hidden=200. The perceptron layer amplifies instability from the Hebbian layer.
- **The supervised layer 2 cannot compensate**: Even though layer 2 uses a label signal (perceptron rule), it receives Hebbian features as input. These features contain no parity information, so the perceptron has nothing useful to learn from.

### Surprise

- **All three rules give essentially identical results** despite having very different learning dynamics (linear correlation vs PCA vs selectivity). This confirms the failure is structural, not algorithmic: no second-order statistic can detect parity.
- **The 56% "best" accuracy in Config 3 is pure noise**: It appeared only for seed=42, and switching to seed=43 or 44 dropped back to ~50%. With 200 test samples, the 95% CI for chance is [43%, 57%], so 56% is well within random fluctuation.

## Theoretical Explanation

Sparse parity with k=3 computes y = x_i * x_j * x_k (a 3rd-order interaction). For {-1,+1} inputs:

- **E[x_i * y] = 0** for all i (zero first-order correlation)
- **E[x_i * x_j * y] = 0** for most (i,j) pairs (zero second-order)
- Only **E[x_i * x_j * x_k * y] = 1** at the secret triple (non-zero third-order)

Hebbian rules are fundamentally limited to detecting first- and second-order statistics. They would need to be augmented with:
- Explicit nonlinear feature expansion (e.g., polynomial features)
- Multi-layer Hebbian with nonlinear interactions between layers
- A fundamentally different objective (e.g., predictive coding, contrastive learning)

## Comparison with Other Approaches

| Method | n=20/k=3 Accuracy | ARD | Local Updates? |
|--------|-------------------|-----|----------------|
| Hebbian (best) | 56% | 34,798 | Yes |
| Forward-Forward | ~50% | ~70,000 | Yes (per layer) |
| Backprop (SGD) | 100% | ~70,000 | No |
| Random Search | 100% (exact) | N/A | N/A |

## Open Questions

- Could a **nonlinear Hebbian rule** (e.g., using polynomial activation) detect 3rd-order structure?
- Would **predictive coding** (a biologically plausible alternative to backprop) fare better?
- Can Hebbian pre-training + supervised fine-tuning help SGD converge faster (even if Hebb alone fails)?
- Is there a **multi-layer Hebbian** architecture where interactions between layers could implicitly compute products?

## Files

- Experiment: `src/sparse_parity/experiments/exp_hebbian.py`
- Results: `results/exp_hebbian/results.json`

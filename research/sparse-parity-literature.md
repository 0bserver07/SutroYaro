# Sparse Parity Learning: Literature Review

**Date**: 2026-03-04
**Context**: Sutro Group Challenge #1 — our 20-bit (k=3) pipeline gets 54%. This review surveys what's known about how to solve it.

## The Problem

Learn a k-sparse parity function: given x ∈ {-1,+1}^n, predict the product of k secret coordinates. With n=20 and k=3, there are C(20,3) = 1,140 possible subsets. The task is statistically easy (O(n^k) samples suffice) but computationally hard (SQ lower bound is Ω(n^k) queries).

## Key Papers

### 1. Hidden Progress in Deep Learning (Barak et al., NeurIPS 2022)
**Paper**: https://arxiv.org/abs/2207.08799

The foundational result. Shows that SGD on neural networks CAN learn sparse parity, but with a sharp phase transition:

- Training loss/accuracy show NO progress for a long time, then suddenly snap to perfect generalization
- The "hidden progress" is SGD gradually amplifying Fourier coefficients corresponding to the secret subset
- Progress is invisible to standard metrics but visible in ||w_t - w_0||_1 (weight movement norm)
- Convergence requires ~n^O(k) iterations, matching SQ lower bounds

**Key hyperparameters from the paper**:
- 1-layer ReLU MLP, hidden=1000
- SGD with LR=0.1, batch_size=32, weight_decay=0.01
- Constant learning rate (no schedule)
- Hinge loss
- Must run for MANY iterations — patience is critical

**Why our pipeline fails**: We use LR=0.5 (5x too high), batch_size=1, and only 10,000 steps. The phase transition may need 50,000+ steps with correct hyperparams.

### 2. Matching the SQ Lower Bound with Sign SGD (Kou et al., NeurIPS 2024)
**Paper**: https://arxiv.org/abs/2404.12376

Proves that Sign SGD (replace gradient with its sign) on 2-layer nets solves k-sparse parity with sample complexity Õ(d^{k-1}), matching the SQ lower bound.

- Uses 2^Θ(k) neurons (for k=3, that's ~8 neurons — much smaller than our 1000)
- Sign SGD normalizes gradient magnitudes, helping with sparse feature detection
- Theoretically optimal, practical implementation straightforward

**Implementation**: Replace `W -= lr * grad` with `W -= lr * sign(grad)` in backward pass.

### 3. Grokking and Sparse Parity (Merrill et al., ICLR 2023)
**Paper**: https://arxiv.org/abs/2303.11873

Sparse parity is a canonical example of "grokking" — delayed generalization after apparent memorization:

- Model first memorizes training data (high train acc, low test acc)
- After many more iterations, suddenly generalizes (test acc jumps to 100%)
- The transition is a competition between "dense" (memorizing) and "sparse" (generalizing) subnetworks
- Weight decay slowly kills the dense subnetwork, allowing the sparse one to dominate

**Implications**: Our 50-epoch run is likely in the memorization phase. Need 500+ epochs.

### 4. GrokFast: Accelerated Grokking (Lee et al., 2024)
**Code**: https://github.com/ironjr/grokfast

A simple trick to accelerate grokking by 50-100x:

- Maintain exponential moving average of gradients: `g_slow = α * g_slow + (1-α) * grad`
- Amplify slow component: `grad_new = grad + λ * g_slow`
- Default α=0.98, λ=2.0
- The slow gradient component corresponds to the generalizing direction
- Low-pass filtering amplifies this signal, dramatically speeding up the phase transition

**This is our most promising practical trick** — could reduce training from 500 epochs to 5-50.

### 5. Feature Learning Dynamics under Grokking (2024)
**Paper**: https://openreview.net/forum?id=gciHssAM8A

Analyzes grokking in sparse parity through the Neural Tangent Kernel (NTK):

- During memorization: NTK eigenfunctions NOT aligned with predictive features
- At generalization onset: NTK top eigenfunctions evolve to focus on the secret indices
- This transition is the mechanism behind the phase transition

### 6. Grokking as Phase Transition (Rubin et al., 2026)
**Paper**: https://arxiv.org/html/2603.01192

Very recent (March 2026) — uses Singular Learning Theory to formalize grokking as a first-order phase transition with a mixed phase.

## Practical Summary

### What MUST change in our pipeline

| Parameter | Current | Target | Why |
|-----------|---------|--------|-----|
| Learning rate | 0.5 | 0.1 | Barak et al. default; too high = overshoot |
| Batch size | 1 | 32 | Reduces gradient noise, standard in literature |
| Max epochs | 50 | 500+ | Phase transition needs n^O(k) steps |
| Training samples | 200 | 500-1000 | More data helps generalization |
| Weight decay | 0.01 | Sweep 0.01-2.0 | Higher WD speeds grokking |

### Novel approaches to try (priority order)

1. **GrokFast** — simplest intervention, 50-100x speedup on grokking
2. **Sign SGD** — theoretically optimal, easy to implement
3. **Cross-entropy loss** — continuous gradients vs hinge saturation
4. **Hidden progress tracking** — ||w_t - w_0||_1, Fourier coefficients

### What probably WON'T help

- Making the network bigger (Barak shows convergence plateaus at large model sizes)
- Fancy optimizers like Adam (SGD is fine, the bottleneck is information-theoretic)
- Data augmentation (parity is invariant to input permutation, but the secret bits aren't)

## Open Questions for Sprint 2

1. Can GrokFast + correct hyperparams solve 20-bit in <1 minute?
2. Does Sign SGD work in pure Python (no numpy) at reasonable speed?
3. What's the energy (ARD) profile during the grokking phase transition?
4. Is the per-layer update scheme compatible with GrokFast?

# Experiment SethTS-005: NoProp for Sparse Parity

**Date**: 2026-04-10 (updated 2026-04-10 with FF baseline)
**Status**: MIXED — NoProp beats Forward-Forward; both beaten by SGD+Curriculum
**Answers**:
1. Does denoising (NoProp) help vs contrastive (FF)? **Yes — significantly.**
2. Does NoProp outperform SGD+Curriculum? **No.**

## Hypothesis

If we train each layer independently as a denoiser (NoProp), then (a) the denoising objective will find k-th order interactions more efficiently than SGD's hinge loss, and (b) eliminating the backward pass across layers will reduce per-step data movement.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 50 |
| k_sparse | 3, 5 |
| hidden | 200 |
| lr (NoProp) | 0.05 (MSE gradients ~2x hinge, so half of SGD's 0.1) |
| lr (SGD) | 0.1 |
| lr (FF) | 0.01 |
| FF threshold | 2.0 |
| wd | 0.01 |
| batch_size | 32 |
| max_epochs | 500–1000 |
| n_train | 2000 |
| seeds | [42, 43, 44, 45, 46] |
| NoProp | T=5 layers, noise=[0.9, 0.7, 0.5, 0.3, 0.1] |
| FF | 2-layer contrastive, goodness = sum(h^2), soft BCE loss |

## NoProp Adaptation

- Target y ∈ {-1,+1} noised by flipping with probability t
- T=5 independent layers, noise schedule [0.9, 0.7, 0.5, 0.3, 0.1] (high → low)
- Each layer receives [x | noisy_y] (n+1 inputs), trains with local MSE loss against clean y
- No gradients flow between layers
- Inference: chain layers sequentially starting from y=0, each refining the prediction

## FF Adaptation

- Label embedded as first input: x_lab = [y | x] (n+1 inputs)
- 2-layer network; positive pass (correct label) + negative pass (wrong label)
- Goodness = sum of squared ReLU activations per layer
- Soft contrastive loss: -log(sigmoid(sign × (goodness − threshold)))
- Hinton normalization between layers (h1/||h1||)

## Results

| Regime | SGD | NoProp | SGD+Curriculum | NoProp+Curriculum | FF | FF+Curriculum |
|---|---|---|---|---|---|---|
| n=20/k=5 | 100%, 70 ep | 100%, 90 ep | 100%, 25 ep | 100%, 33 ep | 40%, 464 ep | 80%, 428 ep |
| n=50/k=3 | 100%, 58 ep | 100%, 42 ep | 100%, 10 ep | 100%, 8 ep | 100%, 73 ep | 100%, 14 ep |
| n=50/k=5 | 0% | 0% | 100%, 34 ep | 100%, 42 ep | 0% | 80%, 310 ep |

**Per-step ARD** (n=20, hidden=200):
- SGD: 8,747
- NoProp: 9,145 (1.05x SGD — slightly more expensive)
- FF: 89,232 (10.2x SGD — 2 passes of [label|x] through 200×200 W2)

**Total DMD on n=50/k=5** (epochs × per-step ARD, curriculum variants):
- SGD+Curriculum: ~297K (34 ep × 8,747)
- NoProp+Curriculum: ~1.92M (42 ep × 5 layers × 9,145) — 6.5x SGD+C
- FF+Curriculum: ~27.7M (310 ep × 89,232) — 93x SGD+C, and only 80% solve rate

## Analysis

### NoProp vs FF (Yaro's question: "Does denoising objective help at all?")

**Yes — NoProp is substantially better than FF on sparse parity.**

- **n=20/k=5**: NoProp 100% vs FF 40% (FF fails more than half the time)
- **n=50/k=3**: NoProp 42ep vs FF 73ep (NoProp ~1.7x faster)
- **n=50/k=5**: NoProp+C 100%/42ep vs FF+C 80%/310ep (NoProp+C clearly better)
- **DMD**: NoProp 1.05x SGD; FF 10.2x SGD. NoProp is ~10x cheaper per step than FF.

**Why NoProp beats FF**: NoProp gives each layer the clean label as a regression target (with noise). The label is directly available to every layer, so all layers optimize toward the same signal. FF only embeds the label as one input dimension — the network must learn to route label information through the goodness objective, which is harder and requires more steps. Additionally, FF's W2 is 200×200 = 40k parameters (label dimension flows through both layers), vs NoProp's W2 is 1×200, making FF's per-step ARD ~10x worse.

### NoProp vs SGD+Curriculum (original negative result — unchanged)

NoProp is still worse than SGD+Curriculum overall:
- Per-step ARD is similar (NoProp 1.05x SGD)
- But NoProp does 5 layer-passes per epoch, making total DMD ~6.5x worse
- Curriculum alone explains the k=5 breakthrough; NoProp adds nothing over SGD once both have curriculum

### What worked
- NoProp is a viable learning algorithm, solves what SGD solves
- **NoProp is clearly better than FF on sparse parity** — both accuracy and DMD
- NoProp+Curriculum solves n=50/k=5 (100%), FF+Curriculum only reaches 80%

### What didn't work
- Neither NoProp nor FF breaks the k=5 wall without curriculum
- FF is 10x more expensive per step (large W2 matrix) and slower to converge
- NoProp's total DMD still 6.5x worse than SGD+Curriculum

### Revised Summary

NoProp sits between FF and SGD+Curriculum: better than FF, worse than SGD+Curriculum. The ranking is:
`SGD+Curriculum > NoProp+Curriculum > FF+Curriculum` for both solve rate and total DMD on the hardest regime.

## Why NoProp Beats FF

FF's contrastive objective requires the network to learn to distinguish positive/negative data by goodness alone. For sparse parity, where the relevant signal lives at the k-th order Fourier coefficient, the goodness function (sum of squared activations) is not well-suited to detecting high-order interactions. NoProp's denoising objective directly regresses y given [x|noisy_y], which is a more direct signal. Each NoProp layer sees the true label (with noise) as an input, making gradient descent more guided.

## Why Neither Beats SGD+Curriculum

The bottleneck for both is the n-scaling problem: irrelevant noise dimensions dominate the gradient signal. Curriculum neutralizes this by keeping n small during the critical learning phase. The choice of learning rule (SGD, NoProp, FF) is secondary once curriculum is applied.

## Open Questions

- Does NoProp's advantage over FF hold at larger n? The n=50 data suggests yes.
- FF's 80% solve rate on n=50/k=5 with curriculum — which seeds fail? Is there a pattern?
- FF's poor ARD is partly structural (200×200 W2 for 2 layers vs 1×200 for NoProp). A 1-layer FF would be closer to NoProp architecturally.

## Files

- Experiment: `src/sparse_parity/experiments/exp_noprop.py`
- Results: `results/exp_noprop/results.json`

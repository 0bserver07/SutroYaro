# Learning Guide: Sparse Parity and Energy-Efficient Training

Start here if you're new to this repo.

## The Big Picture

AI training is bottlenecked by energy, and energy is bottlenecked by memory access. A single read from external memory (HBM) costs ~640 picojoules. A read from a local register costs ~5 picojoules, a 128x difference. The brain runs at ~20 Watts using local update rules. Modern GPUs use thousands of watts, largely shuttling data between memory levels.

The Sutro Group's question: **can we reinvent learning algorithms to be energy-efficient from the ground up?**

We use sparse parity as our "drosophila": simple enough to iterate on in seconds, hard enough to expose the real tradeoffs.

## What Is Sparse Parity?

Take n input bits, each randomly +1 or -1. Pick k secret indices. The label is the product of the inputs at those k positions.

Example with n=5, k=2, secret indices = [1, 3]:

```
Input:  [+1, -1, +1, -1, +1]
Label:  (-1) * (-1) = +1    ← product of positions 1 and 3
```

The challenge: learn which k bits matter out of n total, when the other n-k bits are pure noise.

Why this is hard:
- With n=20 and k=3, there are C(20,3) = 1,140 possible secret subsets
- The gradient signal from the 3 real bits is drowned out by 17 noise bits
- Standard SGD needs ~n^k iterations to find the signal (a theoretical lower bound)

Why this is useful:
- Small enough to run in <1 second
- Hard enough to require real algorithmic work
- The memory access patterns directly expose energy efficiency tradeoffs

## What Is Average Reuse Distance (ARD)?

When your code reads a value from memory, ARD measures how many other values were accessed since that value was last written. If ARD is small, the value is probably still in cache (cheap). If ARD is large, it's been evicted and must be fetched from slow external memory (expensive).

```
Write W1 (10,000 floats)     ← clock: 0
Write b1 (1,000 floats)      ← clock: 10,000
... lots of other operations ...
Read W1                        ← clock: 50,000
                                 Reuse distance: 50,000 - 0 = 50,000
```

In backpropagation, parameter W1 is written at initialization, used in the forward pass, then used again in the backward pass, with the entire forward computation in between. This creates large reuse distances for the first layers.

**Weighted ARD** counts each float's distance equally, so a 10,000-float matrix contributes 10,000x more than a scalar.

Our MemTracker class instruments training code to measure this. See `src/sparse_parity/tracker.py`.

## What Is Grokking?

A neural network memorizes training data (high train accuracy, random test accuracy) for many epochs, then suddenly generalizes (test accuracy jumps to near-perfect). This happens because:

1. SGD is slowly amplifying Fourier coefficients corresponding to the secret indices
2. This progress is invisible to loss/accuracy metrics
3. At a critical point, the signal becomes strong enough to dominate, and generalization happens abruptly

You can track "hidden progress" via ||w_t - w_0||_1 (L1 norm of weight change from initialization). This grows steadily even when accuracy is flat.

Paper: [Barak et al. 2022 "Hidden Progress in Deep Learning"](https://arxiv.org/abs/2207.08799)

## Concepts

### Training Variants (by ARD)

**Standard backprop**: Forward pass → store all activations → backward pass → update all parameters. Large ARD because activations from early layers sit in memory through the entire backward pass.

**Fused layer-wise**: Compute Layer 2 gradients → update W2 immediately → compute Layer 1 gradients → update W1 immediately. Gradient buffers consumed right after creation. ~4% ARD improvement.

**Per-layer forward-backward**: Each layer does forward → backward → update before the next layer starts. Parameters stay in cache between use and update. ~9% ARD improvement. Changes the math (gradients computed with already-updated parameters) but converges identically in practice.

**Forward-Forward (Hinton)**: Two forward passes (positive + negative data), no backward pass. Each layer has its own local objective. Sounds energy-efficient but actually has 25x WORSE ARD than backprop for small networks. Requires 4 weight reads per layer per step instead of 2.

### Sign SGD

Replace `W -= lr * gradient` with `W -= lr * sign(gradient)`. Normalizes gradient magnitudes, helping detect sparse features. Theoretically optimal for sparse parity (matches the Statistical Query lower bound). In practice, 2x faster than standard SGD on k=5.

Paper: [Kou et al. 2024](https://arxiv.org/abs/2404.12376)

### Curriculum Learning

Train on easy problems first (small n), then expand the network to handle harder problems (larger n). For sparse parity, this means:
1. Solve n=10/k=3 (fast)
2. Expand W1 with new columns (small random init)
3. Continue training on n=30/k=3 data (secret indices unchanged)
4. Expand again to n=50

This gives 14.6x speedup and cracks problems that direct training can't solve.

## Papers to Read (in order)

1. **[Hidden Progress in Deep Learning](https://arxiv.org/abs/2207.08799)** (Barak et al. 2022). SGD learns sparse parity via hidden Fourier gap amplification. Explains grokking. Start here.

2. **[Matching the SQ Lower Bound with Sign SGD](https://arxiv.org/abs/2404.12376)** (Kou et al. 2024). Sign SGD is theoretically optimal. Implementation is one line of code.

3. **[A Tale of Two Circuits](https://arxiv.org/abs/2303.11873)** (Merrill et al. 2023). Grokking as competition between sparse (generalizing) and dense (memorizing) subnetworks. Explains why weight decay matters.

4. **[GrokFast](https://github.com/ironjr/grokfast)** (Lee et al. 2024). EMA gradient filter to accelerate grokking. We found it counterproductive when hyperparams are already correct, but useful as a concept.

5. **[Bill Daly on Energy in GPUs](https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457)**. The talk that motivated the whole project. Memory access = energy bottleneck.

## Talks and Resources

- [Interactive ARD tutorial](https://ai.studio/apps/eca3f37a-175a-4713-bb17-622b24e17d3a): hands-on reuse distance exploration
- [NotebookLM on sparse parity](https://notebooklm.google.com/notebook/3eb7b53e-168f-409c-a679-2fb009119e2e): background material
- [parity-nn](https://github.com/Tsili42/parity-nn): minimal codebase for sparse parity experiments
- [cybertronai/sutro](https://github.com/cybertronai/sutro): Yaroslav's reference implementation
- [Hinton's Forward-Forward discussion](https://docs.google.com/document/d/1IdXRUhPRoWt8xLH1Y6iRWRx1g9-gbotFiiAnVixJYZY/edit?tab=t.0): group notes from Meeting #2

## Sutro Group Meeting Notes

The group meets every Monday at 18:00 at South Park Commons (380 Brannan St, SF).

- Meeting #1 (19 Jan): Energy-efficient training intro, the "giraffe nerve" analogy, Bill Daly talk
- Meeting #2 (26 Jan): Forward-Forward algorithm deep dive
- Meeting #3 (02 Feb): Joules measuring, Colab and Modal workflows
- Meeting #4 (09 Feb): "From Beauty to Joules"
- Meeting #5 (16 Feb): Karpathy Names task, optimize character prediction for energy
- Meeting #6 (23 Feb): Germaine's presentation, Emmett's pure-Python GPT (2x memory reduction)
- Meeting #7 (02 Mar): Sparse parity challenge launched, Yaroslav Sprint 1

Full notes: [Sutro Group main doc](https://docs.google.com/document/d/1B9867EN6Bg4ZVQK9vI_ZqykZ5HEtMAHJ7zBGGas4szQ/edit?tab=t.0)

## What We Learned (Meta)

1. **Compare against published baselines first.** Our 20-bit failure (54%) was entirely due to wrong hyperparams (LR=0.5 should be 0.1). One arxiv search fixed it.
2. **Iteration speed matters more than algorithm sophistication.** Muon (first optimizer to beat Adam in 10 years) was discovered on 2-second CIFAR runs. We got our loop to 0.12s.
3. **Simple fixes beat clever algorithms.** GrokFast, Forward-Forward, and per-layer updates all gave marginal or negative improvements. Correct hyperparams gave the biggest win.
4. **Record everything.** Failed experiments (GrokFast, Forward-Forward) prevent future researchers from repeating them.
5. **Leave breadcrumbs.** DISCOVERIES.md + structured findings mean any new session can pick up where the last one left off.

See [findings/prompting-strategies.md](findings/prompting-strategies.md) for the detailed prompting playbook.

# Discoveries

> Accumulated knowledge from all experiments. Read this before starting any new experiment.
> Each entry is a proven fact with a source experiment.

## Proven Facts

### Hyperparameters

- **LR=0.1 is critical** (not 0.5). LR=0.5 overshoots and never triggers the phase transition on 20-bit. [exp1]
- **Batch size=32 helps convergence** but doesn't help ARD (see ARD Metric section). [exp1, exp_b]
- **n_train=500 is sufficient** for n=20, k=3. More data helps generalization. [exp1]
- **Weight decay=0.01 works**. Higher WD not yet tested. [exp1]
- **hidden=1000 is fine** for n=20. hidden=500 also works. hidden=2000 is wasteful. [exp1, exp4]

### Grokking / Phase Transition

- **20-bit sparse parity exhibits grokking**: test accuracy is stuck at ~50% for dozens of epochs, then jumps to 99-100% in ~10 epochs. [exp1]
- **Hidden progress is real**: ||w_t - w_0||_1 grows steadily even when accuracy is flat. This is a useful leading indicator. [exp1]
- **With correct hyperparams, grokking is fast**: 5 epochs to 100% with single-sample SGD (LR=0.1, n_train=500). [exp4 baseline]
- **GrokFast is counterproductive** when hyperparams are already correct. It amplifies gradients that don't need amplifying, causing 83x more weight movement and slower convergence. [exp4]

### Training Variants & ARD

- **Per-layer forward-backward converges identically to standard backprop** on both 3-bit and 20-bit. Same accuracy, same speed. [exp_c]
- **Per-layer gives 3.8% ARD improvement** on 20-bit (17,299 vs 17,976). Consistent across scales. [exp_a, exp_c]
- **Fused gives 1.3% ARD improvement** on 20-bit (17,741 vs 17,976). Less than per-layer. [exp_a]
- **W1 dominates ARD at scale**: W1 (n*hidden floats) accounts for ~75% of all float reads. Its reuse distance is fixed regardless of update order. This caps the improvement from operation reordering at ~10%. [exp_a]

### ARD Metric Limitations

- **ARD doesn't model cache**: batch-32 shows 17x higher ARD than single-sample, but would be far better on real hardware because W1 (80KB at float32) fits in L2 cache. [exp_b]
- **Need cache-simulation mode**: a MemTracker that counts hits vs misses given configurable cache size would give more realistic energy estimates. [exp_b]
- **Parameter writes dominate**: batch-32 does 16x fewer parameter writes than 32 single-sample steps. This is the real energy win. [exp_b]

### Forward-Forward Algorithm

- **FF solves 3-bit parity** perfectly (100% in 4 epochs). [exp_e]
- **FF fails on 20-bit parity** (58.5% max). Greedy layer-wise learning can't coordinate multi-layer feature extraction for sparse parity. [exp_e]
- **FF has 25x WORSE ARD than backprop**. Two forward passes per sample + 4 weight reads per layer per step. The "local learning" advantage is illusory for 2-layer networks. [exp_e]
- **FF might help for 10+ layer networks** where backprop's activation storage creates genuinely large reuse distances. Not our regime. [exp_e]

### Scaling

- **Standard SGD breaks at n^k > 100,000 steps**. Confirmed experimentally. [exp_d]
- **k=3 works up to n≈30-45**. n=30 solved (94.5%), n=50 failed (54%). [exp_d]
- **k=5 is categorically impractical** with standard SGD. Even n=20/k=5 needs ~200,000 epochs. [exp_d]
- **The frontier for novel algorithms is k≥5 or n≥50**. Below that, just tune hyperparams. [exp_d]

## Open Questions (prioritized)

### High Priority
1. **Can Sign SGD solve k=5?** Theory says it needs n^{k-1} samples instead of n^k. For n=20/k=5, that's 160,000 vs 3,200,000. Could be feasible. [ref: Kou et al. 2024]
2. **What does ARD look like with a cache model?** Adding cache simulation to MemTracker would give realistic energy numbers. Batch training would look much better. [from exp_b]
3. **Can curriculum learning help at scale?** Train on easy configs (small n) first, then increase n. Transfer the learned feature detector.

### Medium Priority
4. **Does per-layer + batching combine?** We've tested per-layer with single-sample and batching with standard backprop. What about per-layer + batch=32?
5. **Weight decay sweep**: Does higher WD (0.1, 1.0) accelerate grokking on 20-bit? Would reduce epochs needed.
6. **Tiled/blocked W1 updates**: Since W1 dominates ARD, can we tile the update to keep blocks in cache?

### Exploratory
7. **FF on deeper networks**: Does FF's ARD advantage appear with 5-10 layer networks on a simpler task?
8. **Predictive Coding on sparse parity**: Another local learning rule. Different from FF.
9. **Karpathy Names task**: Separate homework from Meeting #5. Untouched.

## Experiment Log

| ID | Date | Hypothesis | Result | Key Number |
|----|------|-----------|--------|------------|
| exp1 | 03-03 | Fix hyperparams → phase transition | SUCCESS: 99% at epoch 52 | LR=0.1, batch=32 |
| exp4 | 03-03 | GrokFast accelerates grokking | FAILED: baseline SGD faster (5 vs 12 epochs) | GrokFast counterproductive |
| exp_a | 03-04 | Per-layer still wins ARD at scale | CONFIRMED: 3.8% improvement | ARD 17,299 vs 17,976 |
| exp_b | 03-04 | Batch improves ARD | SURPRISE: ARD metric can't capture it | Need cache model |
| exp_c | 03-04 | Per-layer converges on 20-bit | CONFIRMED: 99.5%, identical to standard | Same speed, better ARD |
| exp_d | 03-04 | Find scaling frontier | MAPPED: breaks at n^k > 100K | k=5 impractical |
| exp_e | 03-04 | FF has lower ARD | REFUTED: 25x worse ARD | FF not suitable |
| exp_f | 03-04 | Document prompting strategies | DONE | Literature→diagnose→fix |

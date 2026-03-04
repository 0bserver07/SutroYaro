# Discoveries

> Accumulated knowledge from all experiments. Read this before starting any new experiment.
> Each entry is a proven fact with a source experiment.

## Proven Facts

### Hyperparameters

- **LR=0.1 is critical** (not 0.5). LR=0.5 overshoots and never triggers the phase transition on 20-bit. [exp1]
- **Batch size=32 helps convergence** but doesn't help ARD (see ARD Metric section). [exp1, exp_b]
- **n_train=500 is sufficient** for n=20, k=3. More data helps generalization. [exp1]
- **Weight decay=0.01 is optimal**. Only WD in [0.01, 0.05] solves 20-bit; WD<0.01 too weak (no grokking in 200 epochs), WD>=0.1 too strong (kills learning). The effective regularization LR*WD must be in [0.001, 0.005]. [exp1, exp_wd_sweep]
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

### Per-layer + Batching

- **Per-layer + batch=32 converges identically** to standard+batch. Both solve 5/5 seeds on n=20/k=3. Per-layer needs slightly fewer epochs (40.6 vs 41.4), negligible difference. [exp_perlayer_batch]
- **Single-sample SGD is 8x faster in epochs than batch=32** (5.2 vs ~41 epochs). Batching's value is stability, not convergence speed for this problem. [exp_perlayer_batch]
- **Per-layer + batch has 3.7x wall-time overhead** due to re-forward pass after updating W1. The ARD benefit may offset this on real hardware but was not measured here. [exp_perlayer_batch]

### Scaling

- **Standard SGD breaks at n^k > 100,000 steps**. Confirmed experimentally. [exp_d]
- **k=3 works up to n≈30-45**. n=30 solved (94.5%), n=50 failed (54%). [exp_d]
- ~~**k=5 is categorically impractical** with standard SGD.~~ CORRECTED: k=5 works with enough data. n=20/k=5 solves at 100% with n_train=5000, 14 epochs. The earlier exp_d failure was due to insufficient training data (n_train=200), not the algorithm. [exp_d, exp_sign_sgd]
- **The frontier for novel algorithms is k≥5 or n≥50**. Below that, just tune hyperparams. [exp_d]

### Sign SGD

- **Sign SGD solves k=5 2x faster than standard SGD** (7 vs 14 epochs to 90% with n_train=5000). Both reach 100%. [exp_sign_sgd]
- **Sign SGD with more data converges extremely fast**: 2 epochs to 90% with n_train=20K on k=5. [exp_sign_sgd]
- **Sign SGD needs lr=0.01** (not 0.1). Fixed step size means the lr effectively controls step magnitude directly; lr=0.1 works but lr=0.01 is more stable. lr=0.001 is too slow. [exp_sign_sgd]
- **Sign SGD oscillates near 100%** on k=3 due to fixed step size. Reaches 99% but can't always close the gap. Learning rate decay would likely fix this. [exp_sign_sgd]
- **The n^k sample complexity bound is pessimistic**: Standard SGD solves n=20/k=5 with only 5,000 samples, far below n^k=3,200,000. The practical frontier is much further than theory predicts. [exp_sign_sgd]

### Curriculum Learning

- **n-curriculum demolishes the n^k scaling wall**: Training on n=10 first then expanding W1 to n=50 solves in 20 total epochs vs 292 for direct training — **14.6x speedup**. [exp_curriculum]
- **Transfer after W1 expansion is instant**: After training on n=10/k=3 (18 epochs), expanding to n=20 or n=50 achieves >95% in epoch 1. The learned feature detector for the secret bits transfers perfectly. [exp_curriculum]
- **n-curriculum solves n=50/k=3** which previously failed at 54% in 200 epochs (exp_d). Curriculum bypasses the grokking plateau entirely. [exp_curriculum]
- **k-curriculum gives modest speedup (1.5x)**: Going k=2→k=3→k=5 on n=20 takes 157 epochs vs 232 direct. The k=5 phase still dominates because the parity structure changes. [exp_curriculum]
- **n-curriculum >> k-curriculum**: Scaling input dimension transfers much better than scaling parity order. The hard part is finding which bits matter, and that's invariant to n. [exp_curriculum]

## Open Questions (prioritized)

### High Priority
1. ~~**Can Sign SGD solve k=5?**~~ ANSWERED — Yes, Sign SGD solves k=5 2x faster (7 vs 14 epochs). But standard SGD also solves k=5 with n_train=5000. The real bottleneck was training data, not the optimizer. [exp_sign_sgd]
2. **What does ARD look like with a cache model?** Adding cache simulation to MemTracker would give realistic energy numbers. Batch training would look much better. [from exp_b]
3. ~~**Can curriculum learning help at scale?**~~ ANSWERED — n-curriculum gives 14.6x speedup on n=50/k=3. Transfer is instant after W1 expansion. [exp_curriculum]

### Medium Priority
4. ~~**Does per-layer + batching combine?**~~ ANSWERED — Yes, converges identically. But single-sample is 8x faster in epochs; per-layer+batch adds 3.7x wall-time overhead from re-forward. [exp_perlayer_batch]
5. ~~**Weight decay sweep**~~: ANSWERED — WD=0.01 is optimal, higher WD kills learning. [exp_wd_sweep]
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
| exp_wd_sweep | 03-04 | Higher WD accelerates grokking | REFUTED: WD=0.01 optimal | Only [0.01, 0.05] works |
| exp_curriculum | 03-04 | Curriculum learning helps scaling | SUCCESS: 14.6x speedup | n=50 solved in 20 epochs |
| exp_perlayer_batch | 03-04 | Per-layer + batch combine? | CONFIRMED: converges, but 3.7x slower wall-time | 40.6 vs 41.4 epochs |
| exp_sign_sgd | 03-04 | Sign SGD solves k=5 | SUCCESS: 2x faster, but std SGD also works w/ data | 7 vs 14 epochs to 90% |

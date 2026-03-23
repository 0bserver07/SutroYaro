# Discoveries

> Accumulated knowledge from all experiments. Read this before starting any new experiment.
> Each entry is a proven fact with a source experiment.
> **This is the shared knowledge base.** Anyone can add findings via PR. Format: one bullet, state the fact, cite the source.

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
- **GrokFast helps when k is large**: On n=20/k=5, aggressive GrokFast (a=0.98, l=2.0) gives 2.5x fewer epochs (29 vs 73) and 2.3x faster wall time than SGD. The EMA accumulates the exponentially weak k-th order gradient signal. But on n=30/k=3, it hurts (40% solve rate) because it amplifies noise dimensions. The critical variable is interaction order, not input dimension. [exp_grokfast_v2]
- **GrokFast + curriculum compound**: The two methods are orthogonal (curriculum handles n-scaling, GrokFast handles k-th order plateau). Combined: 5.8x speedup on n=20/k=5, 8.3x on n=50/k=3, and solves n=50/k=5 in 14 epochs / 77ms where SGD completely fails (0% at 1000 epochs). Curriculum shields GrokFast from the noise-dimension problem by keeping n small during the critical learning phase. [exp_grokfast_curriculum]

### Training Variants & ARD

- **Per-layer forward-backward converges identically to standard backprop** on both 3-bit and 20-bit. Same accuracy, same speed. [exp_c]
- **Per-layer gives 3.8% ARD improvement** on 20-bit (17,299 vs 17,976). Consistent across scales. [exp_a, exp_c]
- **Fused gives 1.3% ARD improvement** on 20-bit (17,741 vs 17,976). Less than per-layer. [exp_a]
- **W1 dominates ARD at scale**: W1 (n*hidden floats) accounts for ~75% of all float reads. Its reuse distance is fixed regardless of update order. This caps the improvement from operation reordering at ~10%. [exp_a]

### ARD Metric & Cache Model

- **ARD doesn't model cache**: batch-32 shows 17x higher ARD than single-sample, but would be far better on real hardware because W1 (80KB at float32) fits in L2 cache. [exp_b]
- **CacheTracker built**: LRU cache simulation extends MemTracker. Configurable cache size, tracks hits/misses/hit_rate/effective_ard. [exp_cache_ard]
- **L2 cache (256KB) eliminates ALL misses** for both single-sample and batch, at hidden=200 and hidden=1000. When cache fits the working set, raw ARD is irrelevant — all accesses are hits. [exp_cache_ard]
- **Single-sample is MORE L1-cache-friendly than batch**: batch per-sample temporaries (h_pre_0..31, h_0..31, etc.) thrash L1, giving 69-73% hit rate vs single-sample's 91-100%. [exp_cache_ard]
- **Batch's real advantage is total traffic reduction**: 13% fewer total floats (2.13M vs 2.46M at hidden=1000) and 16x fewer parameter writes. This, not cache hit rate, is the energy win. [exp_b, exp_cache_ard]

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

### Egalitarian Gradient Descent (EGD)

- **EGD halves the grokking plateau on parity**: 14 epochs to 90% vs SGD's 36 (2.6x fewer), 19 to solve vs 42 (2.2x fewer). SVD-normalizes gradients so all directions evolve at equal speed. lr=0.1 works for both. [exp_egd]
- **EGD does not break 10ms**: SVD overhead per batch (~0.12ms) outweighs the epoch savings. GPU wall time 12% worse than SGD despite 2x fewer epochs (1,207ms vs 1,068ms on L4). [exp_egd]
- **EGD is robust to gradient scale**: on sparse sum with MSE loss, SGD at lr=0.1 diverges (0/5 seeds) because gradients scale with target range [-3,3]. EGD solves 5/5 because SVD normalization removes magnitude. [exp_egd]
- **Small hidden (50) is capacity-limited for both**: neither EGD nor SGD solves parity with hidden=50/n_train=500. The optimizer cannot compensate for insufficient capacity. [exp_egd]

### Curriculum Learning

- **n-curriculum demolishes the n^k scaling wall**: Training on n=10 first then expanding W1 to n=50 solves in 20 total epochs vs 292 for direct training — **14.6x speedup**. [exp_curriculum]
- **Transfer after W1 expansion is instant**: After training on n=10/k=3 (18 epochs), expanding to n=20 or n=50 achieves >95% in epoch 1. The learned feature detector for the secret bits transfers perfectly. [exp_curriculum]
- **n-curriculum solves n=50/k=3** which previously failed at 54% in 200 epochs (exp_d). Curriculum bypasses the grokking plateau entirely. [exp_curriculum]
- **k-curriculum gives modest speedup (1.5x)**: Going k=2→k=3→k=5 on n=20 takes 157 epochs vs 232 direct. The k=5 phase still dominates because the parity structure changes. [exp_curriculum]
- **n-curriculum >> k-curriculum**: Scaling input dimension transfers much better than scaling parity order. The hard part is finding which bits matter, and that's invariant to n. [exp_curriculum]

## Open Questions (prioritized)

### High Priority
1. ~~**Can Sign SGD solve k=5?**~~ ANSWERED — Yes, Sign SGD solves k=5 2x faster (7 vs 14 epochs). But standard SGD also solves k=5 with n_train=5000. The real bottleneck was training data, not the optimizer. [exp_sign_sgd]
2. ~~**What does ARD look like with a cache model?**~~ ANSWERED — CacheTracker built. L2 eliminates all misses for both methods. Single-sample is actually more L1-friendly. Batch wins on total traffic (13% fewer floats, 16x fewer writes), not cache locality. [exp_cache_ard]
3. ~~**Can curriculum learning help at scale?**~~ ANSWERED — n-curriculum gives 14.6x speedup on n=50/k=3. Transfer is instant after W1 expansion. [exp_curriculum]

### Medium Priority
4. ~~**Does per-layer + batching combine?**~~ ANSWERED — Yes, converges identically. But single-sample is 8x faster in epochs; per-layer+batch adds 3.7x wall-time overhead from re-forward. [exp_perlayer_batch]
5. ~~**Weight decay sweep**~~: ANSWERED — WD=0.01 is optimal, higher WD kills learning. [exp_wd_sweep]
6. **Tiled/blocked W1 updates**: Since W1 dominates ARD, can we tile the update to keep blocks in cache?

### Blank Slate Approaches

- **Random search solves all tested configs (n≤50, k≤5)**: Enumerate random k-subsets, check exact parity match. O(C(n,k)) tries. n=20/k=3 in 881 tries/0.011s, n=50/k=3 in 11,291 tries/0.142s, n=20/k=5 in 18,240 tries/0.426s. [exp_evolutionary]
- **Random search solves n=50/k=3 which SGD cannot**: Direct SGD gets 54% on n=50/k=3; random search solves it in 0.14s. Combinatorial search bypasses grokking entirely. [exp_evolutionary]
- **Evolutionary search uses fewer evaluations but more wall time**: 18 gens vs 881 tries (n=20/k=3), but evaluating a population of 100 per gen makes it slower in wall time. [exp_evolutionary]
- **Random search and SGD solve different problems**: Random search finds the exact subset (needs enough data to verify); SGD learns a neural net that generalizes (needs grokking). For small k, random search is simpler and more reliable. [exp_evolutionary]
- **Pairwise/greedy feature selection provably fails for parity**: E[y * x_i * x_j] = 0 for ALL pairs including correct ones. Parity is invisible to any correlation test below order k. Same for greedy (single-bit signal is zero). [exp_feature_select]
- **Exhaustive combo search gives 178x–1203x ops speedup over SGD**: Test all C(n,k) subsets with product classifier. 100% correct on n=20/k=3, n=50/k=3, n=20/k=5. Solves n=50/k=3 (0.13s) which SGD fails. [exp_feature_select]
- **Exhaustive scales as O(C(n,k))**: Feasible for k≤7. n=100/k=5 is ~75M combos (minutes). Intractable for k≥10 (C(100,10)=17T). SGD's implicit search wins for large k. [exp_feature_select]
- **Fourier/Walsh-Hadamard solver is 13x faster than SGD on n=20/k=3** (0.009s vs 0.12s). Computes mean(y * prod(x[:,S])) for each k-subset — true subset has correlation 1.0, all others ~0. 100% accuracy on every config tested. [exp_fourier]
- **Fourier needs only 20 samples for k=3** vs 500-5000 for SGD. Sample complexity is O(1/epsilon^2), independent of n. [exp_fourier]
- **Fourier solves n=200/k=3 (1.3M subsets) in 10.8s and n=20/k=7 (77K subsets) in 0.7s**. Scales as O(C(n,k) * n_samples). [exp_fourier]
- **Fourier ARD is 64x worse than SGD** (1,147,375 vs 17,976). Pure streaming over data for each subset — no weight reuse, no locality. [exp_fourier]

### Algebraic / Exact Methods

- **GF(2) Gaussian elimination solves in ~500μs** and is k-independent. Works for k=3,5,7,10 equally fast. Only needs n+1 samples (21 for n=20). 240x faster than SGD. [exp_gf2]
- **GF(2) is fragile to noise**: basic solver fails at 1% noise (inconsistent system). But subset-sampling robust solver recovers up to 10-15% noise by finding clean equation subsets. 100% success at 10% noise, 65% at 15%, 20% at 20%. [exp_gf2_noise]
- **Kushilevitz-Mansour finds secret via influence estimation in O(n)** not O(C(n,k)). ARD of 1,585 vs Fourier's 1,147,375 (724x better). Even 5 influence samples per bit suffice. [exp_km]
- **SMT/backtracking constraint solver at 0.002s**. The k-1 pruning trick: once k-1 indices fixed, last column is fully determined. Only 10 samples needed. [exp_smt]

### Information-Theoretic Methods

- **MI provides no advantage over Fourier for binary parity**: 3.7x slower (0.033s vs 0.009s), same ARD. True subset MI is ~0.693 nats (log 2), wrong subsets ~0.001. [exp_mutual_info]
- **LASSO on interaction features is competitive**: 0.005s, robust to alpha across 500x range. LASSO finds exactly 1 nonzero coefficient. Same combinatorial cost as Fourier (O(C(n,k))). [exp_lasso]
- **MDL is noise-robust**: works under 5% label noise. True subset compresses to 0 bits, wrong subsets ~499 bits. 30% slower than Fourier. [exp_mdl]
- **Random projections save 30-70% of evaluations** vs exhaustive Fourier but with high variance (geometric distribution). Modest wall-time speedup (1.1-1.5x). [exp_random_proj]

### Local Learning Rules (all failed)

- **All local learning rules fail on sparse parity**: Hebbian (~50%), Predictive Coding (~51-55%), Equilibrium Propagation (~60%), Target Propagation (~55%). This is structural, not tunable. [exp_hebbian, exp_predictive_coding, exp_equilibrium_prop, exp_target_prop]
- **Parity is invisible to methods limited to local statistics**: the signal lives in k-th order interactions (E[x_i * x_j * x_k * y] = 1). Any method that only detects 1st/2nd-order statistics gets zero signal. [exp_hebbian]
- **Predictive coding has 18x worse ARD than backprop** (370K vs 20K on n=20/k=3). 15 inference iterations re-read weight matrices ~32 times. [exp_predictive_coding]
- **Equilibrium propagation is 2,300x slower than SGD** and fails due to tanh saturation. 60 relaxation iterations (30 free + 30 clamped) per training step. [exp_equilibrium_prop]
- **Target propagation suffers "target collapse"**: the linear inverse G2 produces input-independent targets. ARD is 1.1-1.6x worse than backprop due to extra buffers. [exp_target_prop]

### Hardware-Aware Methods

- **Tiled W1 increases software ARD by 6.8-12.1%**: the MemTracker can't capture hardware cache benefits. The output layer still needs the full hidden vector before backward starts. [exp_tiled_w1]
- **Pebble game optimizer saves 2.2% energy** (10.76 uJ vs 11.00 uJ). Found that fused and per-layer orderings break training (57% accuracy) via read-after-write hazard on mutable parameters. [exp_pebble_game]
- **Binary weights solve n=3/k=3 in 1 epoch (80x faster)** but fail on n=20/k=3 (~55%). STE is too crude for the feature selection problem. [exp_binary_weights]

### Alternative Framings

- **GP evolves exact symbolic solution for n=20/k=3** (e.g., mul(x[0], mul(x[15], x[17]))). Zero parameters, zero ARD. Fails n=50/k=3 and n=20/k=5 due to needle-in-haystack fitness landscape. [exp_genetic_prog]
- **RL bit querying achieves theoretical minimum inference reads**: k reads per prediction, ARD=1. Value-blind state (track which bits queried, not values) was the key to making Q-learning converge. [exp_rl]
- **Decision trees fail on parity**: best is ExtraTrees at 92.5% (n=20/k=3). Greedy information-gain splitting fails because individual bits have zero marginal correlation. [exp_decision_tree]

### DMC (Data Movement Complexity) Metric

- **DMC and ARD rankings disagree**: GF2 wins DMC on parity (8,607) despite KM winning ARD (92). GF2 accesses 5x fewer total floats (860 vs 4,420). DMC = sum(size * sqrt(stack_distance)) penalizes total data movement, not just average distance. [exp_dmc_optimize]
- **KM-min (1 influence sample) achieves DMC 3,578**: 58% below GF2 baseline, 83% below standard KM. Parity influence is deterministic (exactly 0 or 1), so a single sample per bit suffices. All stack distances are 20 floats (fits in L1). 5/5 seeds correct. [exp_dmc_optimize]
- **GF2 harness DMC is artificially low**: the harness tracks only 3 coarse operations (write matrix, read matrix, write solution). With fine-grained tracking of O(n^2) row operations, GF2's true DMC is 189,056 (22x higher than reported 8,607). [exp_dmc_optimize]
- **Fourier DMC is 78 billion**: 9 million times worse than GF2, despite being fast in wall time (0.066s). Reads the full dataset for each of C(20,3)=1,140 subsets. [exp_dmc_optimize]
- **SGD on sparse sum has the lowest DMC of any method on any challenge** (2,862). Sum has first-order structure that gradient descent exploits in 1 epoch. [exp_dmc_optimize]

DMC baseline rankings (sparse parity, n=20, k=3):

| Method | ARD | DMC | Total Floats |
|--------|-----|-----|-------------|
| KM-min (1 sample) | 20 | 3,578 | 1,600 |
| KM-inplace | 30 | 4,319 | 1,200 |
| GF2 (harness) | 420 | 8,607 | 860 |
| KM (5 samples) | 92 | 20,633 | 4,420 |
| SMT | 3,360 | 348,336 | 6,720 |
| SGD | 8,504 | 1,278,460 | 24,470 |
| Fourier | 11,980,500 | 78,140,662,852 | 23,961,000 |

### Exploratory
7. **FF on deeper networks**: Does FF's ARD advantage appear with 5-10 layer networks on a simpler task?
8. ~~**Predictive Coding on sparse parity**~~: ANSWERED — Failed, 18x worse ARD than backprop. Generative model is harder than discriminative for parity. [exp_predictive_coding]
9. **Karpathy Names task**: Separate homework from Meeting #5. Untouched.
10. ~~**Can GF(2) handle noisy labels?**~~ ANSWERED — Basic GF(2) fails at 1% noise (inconsistent). Robust subset-sampling solver works up to 10-15% noise. [exp_gf2_noise]
11. **Hybrid approach: use KM to find candidate bits, then verify with small neural net.** What's the total energy cost?
12. **At what depth does predictive coding's locality advantage over backprop appear?** Our 2-layer network is too shallow.
13. **Can the pebble game optimizer's anti-dependency detection be automated for arbitrary computation graphs?**

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
| exp_cache_ard | 03-04 | Cache model shows batch wins | NUANCED: L2 eliminates all misses; batch wins on traffic not locality | SS 100% L1 hit vs batch 73% |
| exp_sign_sgd | 03-04 | Sign SGD solves k=5 | SUCCESS: 2x faster, but std SGD also works w/ data | 7 vs 14 epochs to 90% |
| exp_evolutionary | 03-04 | Random/evo search over k-subsets | SUCCESS: solves all configs incl n=50/k=3 | Random: 881-18K tries, <0.5s |
| exp_feature_select | 03-04 | Feature selection vs SGD | PARTIAL: exhaustive 178-1203x faster, pairwise/greedy provably fail | Parity invisible below order k |
| exp_fourier | 03-04 | Walsh-Hadamard correlation finds secret | SUCCESS: 13x faster than SGD, 100% on all configs | 0.009s n=20/k=3, ARD 64x worse |
| exp_mutual_info | 03-06 | MI detects parity | SUCCESS but no advantage over Fourier | 0.033s, same ARD |
| exp_lasso | 03-06 | LASSO finds 1 coefficient | SUCCESS: 0.005s, robust | 1 nonzero coef |
| exp_decision_tree | 03-06 | Trees learn parity | FAILED: 92.5% best, greedy fails | Zero marginal signal |
| exp_gf2 | 03-06 | GF(2) solves parity | SUCCESS: ~500μs, k-independent | 240x faster than SGD |
| exp_random_proj | 03-06 | Monte Carlo Fourier | SUCCESS: saves 30-70% evals | High variance |
| exp_km | 03-06 | KM influence estimation | SUCCESS: 0.006s, ARD 1,585 | 724x better ARD |
| exp_hebbian | 03-06 | Hebbian learns parity | FAILED: ~50% (chance) | 3rd-order invisible |
| exp_predictive_coding | 03-06 | PC lower ARD than backprop | FAILED: 18x worse ARD | Generative harder |
| exp_equilibrium_prop | 03-06 | EP solves parity | FAILED: ~60%, 2300x slower | Tanh saturation |
| exp_target_prop | 03-06 | TP local targets work | FAILED: ~55%, target collapse | Input-independent |
| exp_tiled_w1 | 03-06 | Tiling reduces ARD | FAILED: ARD increased 6.8% | SW metric mismatch |
| exp_pebble_game | 03-06 | Optimal execution order | PARTIAL: 2.2% energy win | Fused breaks training |
| exp_binary_weights | 03-06 | Binary ops for parity | PARTIAL: n=3 works, n=20 fails | STE too crude |
| exp_genetic_prog | 03-06 | GP finds symbolic solution | PARTIAL: n=20/k=3 only | Needle in haystack |
| exp_smt | 03-06 | Constraint solver | SUCCESS: 0.002s, 10 samples | k-1 pruning trick |
| exp_rl | 03-06 | RL learns what to read | SUCCESS: k reads per prediction | ARD=1 at inference |
| exp_mdl | 03-06 | MDL finds secret | SUCCESS: noise-robust | 0 bits vs 499 bits |
| exp_gf2_noise | 03-09 | GF(2) handles noise | SUCCESS: robust solver | 100% at 10% noise, fails at 20% |
| exp_egd | 03-16 | EGD eliminates grokking plateau | PARTIAL: 2x fewer epochs but SVD overhead | 14 vs 36 ep to 90%, 12% slower wall |
| exp_dmc_optimize | 03-22 | Reduce DMC below GF2 baseline | SUCCESS: KM-min DMC 3,578 (-58%) | 1 sample suffices, GF2 under-counted |

---

## Challenge 2: Sparse Sum

y = sum of x[secret_indices]. Output in [-k, k]. Regression, not classification.
Unlike parity, each secret bit contributes independently (first-order signal).
This tests whether the infrastructure generalizes to new tasks.

### Baselines (n=20, k=3, seed=42)

| Method | Accuracy | ARD | Time | Notes |
|--------|----------|-----|------|-------|
| SGD (gradient descent) | 100% | 20 | 2.6ms | 1 epoch, tracked steps |
| OLS (least squares) | 100% | 20,980 | 2.5ms | One-shot matrix solve |
| KM influence | 100% | 92 | 3.7ms | Same approach as parity |
| Fourier (first-order) | 100% | 220,500 | 0.5ms | Only checks n bits, not C(n,k) |
| GF(2) | 0% (fails) | -- | -- | Sum is not parity over GF(2) |

### Key differences from parity

- Sum is linear. Parity is k-th order. This is the fundamental distinction.
- SGD solves sum in 1 epoch (vs ~40 for parity) with ARD of 20 (vs 17,976 for parity).
- GF(2) fails on sum because sum is not linear over GF(2).
- Fourier on sum only needs to check n individual bits, not C(n,k) subsets.
- KM has the best ARD for exact methods (92 vs parity's 1,585).

### Open Questions

1. Do local learning rules (Hebbian, Predictive Coding) succeed on sparse sum?
2. How does ARD scale with n for sum vs parity?

---

## Challenge 3: Sparse AND

y = product((x[secret]+1)/2). Maps {-1,+1} to {0,1} per bit, then takes product (logical AND).
Output is 1 only when ALL k secret bits are +1. P(y=1) = 1/2^k.
This is a highly asymmetric classification task — class imbalance grows exponentially with k.

Unlike parity (XOR over {-1,+1}), AND is a conjunction over {0,1}.
Both are k-th order interactions: the signal is invisible to methods that only
detect lower-order statistics. But AND has severe class imbalance that parity does not.

### Baselines (n=20, k=3, seed=42)

| Method | Accuracy | ARD | Time | Notes |
|--------|----------|-----|------|-------|
| SGD (neural net) | 100% | 29,164 | 12.2ms | 4 epochs, sigmoid output |
| KM influence (5 samples) | 81% | 92 | 1.3ms | Too few samples for 1/2^(k-1) signal |
| KM influence (20 samples) | 100% | 367 | 1.3ms | Needs more samples than parity |
| Fourier (exhaustive) | 100% | 11,980,500 | 17.1ms | C(20,3) subset checks |
| GF(2) | 0% (fails) | -- | -- | AND is not linear over GF(2) |

### Key differences from parity

- AND maps to {0,1} vs parity's {-1,+1}. Both are k-th order interactions.
- AND has severe class imbalance: P(y=1) = 1/2^k (12.5% for k=3, 3% for k=5).
- Parity is balanced: P(y=+1) = P(y=-1) = 0.5 regardless of k.
- GF(2) fails on AND because AND is not linear over GF(2).
- KM influence per secret bit is 1/2^(k-1) for AND vs 1 for parity.

### Open Questions

1. Does class imbalance affect SGD convergence on AND vs parity?
2. Does KM need more samples per bit to detect AND influence (1/2^(k-1) vs 1)?
3. How does ARD compare between AND and parity for the same method?

# Proposed Approaches

17 alternative ways to solve sparse parity, beyond the standard SGD and the three blank-slate experiments already run ([Fourier](../findings/exp_fourier.md), [evolutionary](../findings/exp_evolutionary.md), [feature selection](../findings/exp_feature_select.md)).

The core problem: given n-dimensional inputs where only k bits determine the label (via parity), find those k bits and classify correctly. Minimize energy (memory access cost) while doing it.

---

## Information-Theoretic

### 1. Mutual Information Estimation

Estimate MI(y; x_S) for candidate subsets S. Unlike Fourier which computes exact product correlation, MI detects any nonlinear relationship between a subset and the label. Could use binning, kNN (Kraskov estimator), or MINE.

- **Why it might work**: MI is a strictly more general test than Fourier correlation. If the parity function had any noise or was only approximately parity, MI would still detect it while Fourier might miss it.
- **Energy angle**: Streaming MI estimator could process samples one at a time with minimal buffer. ARD depends on the estimator choice.
- **Complexity**: Still O(C(n,k)) subset evaluations. The question is whether each evaluation needs fewer samples than Fourier's ~20.

### 2. LASSO on Interaction Features

Expand input to all C(n,k) interaction terms (x_i * x_j * x_k for k=3), then run L1-penalized linear regression. The true solution has exactly 1 nonzero coefficient — a perfect match for LASSO's sparsity assumption.

- **Why it might work**: LASSO is provably optimal for sparse linear recovery. The parity function IS linear in the interaction basis. With enough samples, LASSO finds the single active interaction term.
- **Energy angle**: LASSO's coordinate descent updates one coefficient at a time — high locality. But the expanded feature matrix has C(n,k) columns, which could be large.
- **Complexity**: O(C(n,k) * n_samples) to construct features, then LASSO convergence. For n=20/k=3, 1,140 features — trivial. For n=50/k=5, 2.1M features — still feasible.

### 3. Decision Trees / Random Forests

A depth-k binary decision tree can learn k-parity exactly. Each split tests one bit. Random forests with max_depth=k would implicitly search the subset space.

- **Why it might work**: Parity is a natural tree problem. A tree of depth 3 testing bits a, b, c with XOR logic at the leaves solves it exactly. Sklearn's `DecisionTreeClassifier(max_depth=k)` might find it directly.
- **Energy angle**: Trees have no weight matrices, no gradient computation, no backprop. Memory access is a single path from root to leaf — minimal ARD.
- **Complexity**: Tree construction is O(n * n_samples * 2^depth) per node. Might struggle because parity doesn't produce clean single-feature splits (each individual bit has zero correlation with the label).
- **Risk**: Greedy splitting by information gain fails for parity (same reason greedy feature selection fails — individual bits have zero MI with the label). Would need interaction-aware splitting or random subspace sampling.

---

## Algebraic / Exact

### 4. Gaussian Elimination over GF(2)

Treat each sample as a linear equation over GF(2) (binary field). Convert real-valued inputs to bits (sign), labels to bits. With ~n samples, solve the binary linear system to find which variables participate in the parity.

- **Why it might work**: Parity IS a linear function over GF(2). Gaussian elimination solves linear systems in O(n^2) per sample. With n+1 linearly independent samples, the solution is exact.
- **Energy angle**: O(n^2) operations total. No iteration, no convergence, no gradients. Memory footprint is one n x n binary matrix. This is the theoretically optimal approach for pure parity.
- **Complexity**: ~20 samples and ~400 bit operations for n=20. Microseconds. The only question is whether it generalizes to noisy/approximate parity.
- **Limitation**: Only works for exact parity (linear over GF(2)). If the true function were nonlinear or noisy, this fails. But for our problem definition, it's the correct tool.

### 5. Random Projections + Correlation

Instead of testing all C(n,k) subsets exhaustively (Fourier approach), project to random k-dimensional subspaces and test correlation there. Monte Carlo version of the Fourier solver.

- **Why it might work**: If you sample random k-subsets, you hit the correct one with probability 1/C(n,k). Each test is O(n_samples). Expected total work is O(C(n,k) * n_samples) — same as Fourier, but with early stopping when you find a high-correlation subset.
- **Energy angle**: Streaming evaluation. Test one random subset, check correlation, move on. No persistent state between evaluations. ARD is minimal.
- **Complexity**: Expected C(n,k) evaluations. For n=20/k=3, ~1140 tries. Similar to the random search in exp_evolutionary (881 tries). The difference is this uses correlation instead of exact match, which is more robust to noise.

### 6. Kushilevitz-Mansour (KM) Algorithm

The KM algorithm learns functions with sparse Fourier spectra in poly(n, 2^k) time. Designed exactly for this problem class. Estimates Fourier coefficients and identifies the large ones.

- **Why it might work**: This is the theoretically principled solution for learning sparse Boolean functions from membership queries. Sparse parity has exactly one large Fourier coefficient (the secret subset).
- **Energy angle**: Query-efficient — uses O(2^k * log(n)) samples. Memory is O(n) for the coefficient estimates. Very low footprint.
- **Complexity**: Poly(n, 2^k). For k=3, this is ~8 * log(20) ≈ 35 queries. Absurdly efficient. For k=10, it's ~1024 * log(n) — still polynomial.
- **Why it matters**: This is the algorithm that PROVES sparse parity is easy in the PAC learning framework. If we haven't tried it, we should — it's the theoretical benchmark.

---

## Local / Energy-Aware Learning

### 7. Hebbian + Anti-Hebbian Learning

Purely local learning rule. Neurons that fire together strengthen connections (Hebb), anti-correlated pairs weaken (anti-Hebb). No backprop, no global error signal. Weight updates depend only on pre- and post-synaptic activity.

- **Why it might work**: Hebbian learning extracts principal components of the input distribution. If the parity-relevant bits produce correlated activity patterns in a nonlinear network, Hebbian learning might isolate them.
- **Energy angle**: Updates are purely local — each weight update uses only its two connected neurons. ARD is minimal by construction. This is the closest to how biological synapses work.
- **Risk**: Standard Hebbian learning finds linear correlations, and parity bits have zero linear correlation with the label. Would need a nonlinear Hebbian variant or multi-layer Hebbian with nonlinearities.

### 8. Predictive Coding

Each layer predicts the layer below, updates based on prediction error. Error signals propagate locally — each layer only talks to its neighbors. Rao & Ballard (1999), updated by Millidge et al. (2021).

- **Why it might work**: Under certain conditions, predictive coding approximates backprop's weight updates. But the memory access pattern is different — each layer updates independently using local prediction errors.
- **Energy angle**: Local error signals mean each layer's working set is small (its own weights + activations + one neighbor). No storing the full forward pass. ARD should be much better than standard backprop.
- **Complexity**: Similar to backprop in compute. The difference is memory access pattern, not total operations.
- **Already flagged**: In DISCOVERIES.md as an open question. Different from Forward-Forward — PC minimizes prediction error, FF maximizes/minimizes goodness.

### 9. Equilibrium Propagation

Scellier & Bengio (2017). The network settles to an equilibrium (free phase), then is nudged toward the correct output (clamped phase). Weight updates use the difference between free and clamped states. Only forward passes — no backprop.

- **Why it might work**: Proven to approximate backprop gradients in the limit of small nudging. Uses only local information (pre- and post-synaptic activity in two phases). Implementable on analog hardware.
- **Energy angle**: Two forward relaxations per sample, no backward pass. Activations from the free phase can be reused in the clamped phase. Memory footprint is one copy of activations.
- **Risk**: Convergence to equilibrium requires multiple iterations per phase. Total compute might exceed backprop. The energy win is in memory access pattern, not total operations.

### 10. Target Propagation

Each layer gets a local target (computed by approximate inverse mappings from the layer above). Layers train with local losses toward their targets. Bengio (2014), Lee et al. (2015).

- **Why it might work**: Avoids backprop's long dependency chain. Each layer trains independently once targets are set. Unlike FF's greedy approach, targets carry information from the global loss.
- **Energy angle**: After target computation (one backward pass of inverses), all weight updates are local. The inverse pass has similar memory access to backprop, but the weight update phase is fully local.
- **Risk**: Learning good inverse mappings is hard. The inverse itself must be trained, creating a chicken-and-egg problem. For a 2-layer network on sparse parity, the overhead might not be worth it.

---

## Hardware-Aware / Scheduling

### 11. Tiled W1 Updates

Since W1 (input-to-hidden weights) dominates ARD at 75% of float reads, tile the weight matrix into blocks that fit in L1 cache. Process each tile's forward/backward/update before moving to the next.

- **Why it might work**: Already identified as the bottleneck in exp_a. W1 is read in forward, then read again much later in backward. Tiling breaks this into smaller chunks that stay in cache.
- **Energy angle**: If each tile fits in L1 (64KB), the reuse distance within a tile drops to near zero. Total operations unchanged, but cache behavior changes drastically.
- **Complexity**: Requires rewriting the forward/backward to operate on weight blocks. For hidden=200, n=20: W1 is 20x200=4000 floats = 16KB at float32. Already fits in L1. Tiling only matters at larger hidden sizes.

### 12. Pebble Game Optimizer

Formalize the computation graph as a pebble game (from Meeting #6, Sethi 1975). Each tensor is a pebble. Placing a pebble on a register costs 5pJ, on L1 costs 20pJ, on HBM costs 640pJ. Find the pebbling schedule that minimizes total energy.

- **Why it might work**: This separates the algorithm design from the scheduling problem. Keep SGD (or any algorithm) fixed, but let an optimizer (or AI agent) find the best execution order.
- **Energy angle**: This is directly optimizing the thing we care about — energy from memory access. The pebble game is the formal model.
- **Complexity**: Optimal pebbling is NP-hard in general, but our computation graphs are small (2 layers, ~10 operations). Exhaustive search over valid schedules might be feasible.
- **Connection**: Michael showed a pebble game implementation at Meeting #7. Andy built a prototype scheduler.

### 13. Mixed Precision / Binary Weights

Train with int8 or binary weights. Sparse parity with sign-encoded inputs might work with binary arithmetic — the inputs are already sign-encoded, and parity is a product of signs.

- **Why it might work**: Parity is fundamentally a binary operation. A network with binary weights (+1/-1) and sign activation functions can represent parity exactly. BinaryConnect (Courbariaux et al., 2015) showed binary networks can learn on MNIST.
- **Energy angle**: Binary operations use ~30x less energy than float32 multiplies. Memory footprint shrinks 32x. L1 cache holds 32x more parameters. ARD drops proportionally.
- **Risk**: Training binary networks is hard — straight-through estimator for gradients. But the problem is simple enough that it might converge anyway.

---

## Alternative Framings

### 14. Genetic Programming

Evolve *programs* (symbolic expressions) that compute parity. The target solution is literally `sign(x[a] * x[b] * x[c])`. GP represents candidates as expression trees and evolves them via crossover and mutation.

- **Why it might work**: The solution is a short symbolic expression. GP excels at finding short programs in a large space. The fitness function (classification accuracy) is fast to evaluate.
- **Energy angle**: The discovered program has zero parameters, zero memory footprint, zero ARD. A symbolic solution is the ultimate energy-efficient algorithm — it's just a formula.
- **Complexity**: GP search space is large but the target expression is short (depth 3-4). Population of ~100, should converge in under 100 generations for k=3.

### 15. Program Synthesis / SMT Solver

Encode the problem as a constraint: find indices a, b, c such that `sign(x[a] * x[b] * x[c]) == label` for all training samples. Feed to Z3 or similar SMT solver.

- **Why it might work**: This is exactly what SMT solvers are designed for — find variable assignments satisfying a set of constraints. The search space is C(n,k), which is tiny for SMT.
- **Energy angle**: The solver runs once, produces the answer, done. No training loop. Energy is whatever the solver uses to search — likely microseconds for n=20/k=3.
- **Complexity**: SMT over integer variables with C(20,3)=1140 possible assignments. Trivial for modern solvers. Interesting question: how does it scale to n=200/k=5?

### 16. Reinforcement Learning

An agent observes samples one at a time. At each step it picks a bit index to "query" (observe the sign of that bit). After k queries, it predicts the label. Reward for correct prediction.

- **Why it might work**: The agent must learn which bits are informative through exploration. This is sparse parity as a sequential decision problem. The optimal policy queries exactly the k secret bits.
- **Energy angle**: Each step reads one value. Total memory access per sample is k reads + 1 prediction. ARD is the distance between reading the k-th bit and predicting — about k steps.
- **Complexity**: RL sample efficiency is the question. The agent must explore all n bits to discover which k matter. With n=20 and k=3, the state space is manageable. Epsilon-greedy or UCB over bit indices.
- **Different lens**: This frames energy-efficient learning as "learning what to look at" — minimum number of memory reads per prediction.

### 17. Minimum Description Length / Compression

The best compressor of the label sequence is the one that knows the secret bits. For each candidate k-subset, compute the compressed length of labels given that subset's parity. The shortest description wins.

- **Why it might work**: MDL is a principled model selection criterion. The true subset compresses labels to ~0 bits (deterministic). Any wrong subset compresses to ~n_samples bits (random).
- **Energy angle**: Streaming compression. Process samples sequentially, maintain a running compression estimate per candidate subset. Memory is O(C(n,k)) counters.
- **Complexity**: O(C(n,k) * n_samples) — same as Fourier. But MDL is more general: it works for any deterministic labeling function, not just parity. If the problem changed from parity to some other Boolean function, MDL still works.

---

## Summary Table

| # | Approach | Type | Expected Speed (n=20/k=3) | ARD Estimate | Works for k≥10? |
|---|----------|------|--------------------------|-------------|-----------------|
| 1 | Mutual Information | Info-theory | ~C(n,k) evals | Low (streaming) | No |
| 2 | LASSO on interactions | Info-theory | Fast (sklearn) | Medium | No (C(n,k) features) |
| 3 | Decision Trees | Info-theory | Fast (sklearn) | Minimal | No (greedy fails) |
| 4 | GF(2) Gaussian Elimination | Algebraic | Microseconds | Minimal (n x n matrix) | Yes |
| 5 | Random Projections | Algebraic | ~C(n,k) evals | Minimal (streaming) | No |
| 6 | Kushilevitz-Mansour | Algebraic | ~35 queries | Minimal | Yes (poly in 2^k) |
| 7 | Hebbian Learning | Local | Unknown | Minimal by construction | Unknown |
| 8 | Predictive Coding | Local | ~backprop | Better than backprop | Yes |
| 9 | Equilibrium Propagation | Local | Slower than backprop | Better than backprop | Yes |
| 10 | Target Propagation | Local | ~backprop | Better than backprop | Yes |
| 11 | Tiled W1 | Hardware | Same as SGD | Much better | Yes |
| 12 | Pebble Game Optimizer | Hardware | Same as SGD | Optimal by construction | Yes |
| 13 | Binary Weights | Hardware | Faster (binary ops) | 32x smaller footprint | Yes |
| 14 | Genetic Programming | Alt framing | ~100 generations | Zero (symbolic solution) | Unlikely |
| 15 | SMT Solver | Alt framing | Microseconds | N/A (one-shot) | Depends on solver |
| 16 | RL (bit querying) | Alt framing | Unknown | k reads per sample | Yes |
| 17 | MDL / Compression | Alt framing | ~C(n,k) evals | Low (streaming) | No |

### Priority Recommendations

**Try first** (likely fast wins):

- **#4 GF(2) Gaussian Elimination** — theoretically optimal, should solve in microseconds
- **#6 Kushilevitz-Mansour** — the PAC-learning benchmark, ~35 queries for k=3
- **#15 SMT Solver** — one-shot, might solve instantly

**Most relevant to the energy question** (the actual research goal):

- **#8 Predictive Coding** — direct backprop alternative with local updates
- **#12 Pebble Game Optimizer** — directly optimizes what we measure
- **#11 Tiled W1** — addresses the known bottleneck from exp_a

**Most creative / different lens**:

- **#14 Genetic Programming** — discovers a zero-parameter symbolic solution
- **#16 RL bit querying** — reframes energy as "what to look at"
- **#17 MDL** — works for any Boolean function, not just parity

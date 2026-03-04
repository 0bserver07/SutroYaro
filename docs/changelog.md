# Changelog

All notable changes to this research workspace.

## [0.8.0] - 2026-03-04

### Blank-slate approaches (Round 3 — no neural nets, no SGD)

The feedback on rounds 1-2 was that all experiments were incremental variations on the same MLP+SGD recipe. Round 3 reframes the problem entirely: sparse parity as a **search problem**, not a learning problem.

**The core insight**: for k=3 parity with secret indices {a,b,c}, the label is simply `x[a] * x[b] * x[c]`. You don't need a neural network to find this — you need to search over C(n,k) possible subsets and test which one matches the data.

- **Fourier / Walsh-Hadamard solver** — Sparse parity IS a Fourier coefficient. For each candidate k-subset S, compute `mean(y * product(x[:, S]))`. The true secret gives correlation ~1.0, everything else gives ~0. No training, no gradients, no iterations. Result: **13x faster than SGD** (0.009s vs 0.12s for n=20/k=3), solves n=50/k=3 and n=20/k=5 trivially where SGD struggles. Only needs 20 samples. Scales to k=7 before combinatorial explosion (C(n,k) subsets).

- **Evolutionary / random search** — Even simpler: randomly sample k-subsets, test if `product(x[:, subset])` matches all labels. For n=20/k=3 it takes ~881 random tries (0.011s). Evolutionary search with mutation+crossover solves it in fewer evaluations but more wall time. Solves n=50/k=3 in 0.14s — a config SGD fails on entirely. No math required, just brute-force enumeration.

- **Feature selection** — Tried to decompose into "find the bits" then "classify." Exhaustive combo search works (178-1203x fewer ops than SGD). But the clever approaches fail: pairwise correlations are provably zero for parity (E[y * x_i * x_j] = 0 for ALL pairs, even correct ones). Greedy forward selection also fails. Parity has **zero low-order statistical signatures** — you must test the full k-way interaction. This is why neural nets need so many iterations: they're implicitly searching through the combinatorial space via gradient descent.

**When to use what**:
- k ≤ 7: Fourier/random search wins (milliseconds, exact, guaranteed)
- k ≥ 10: C(n,k) explodes, SGD's implicit search via gradients becomes the only feasible path
- k = 8-9: hybrid approaches may work (combinatorial search with pruning)

---

## [0.7.0] - 2026-03-04

### Research experiments (Round 2 — autonomous agent team)

Round 2 explored variations on the working SGD solution: different optimizers, hyperparameter sweeps, training curricula, and energy measurement improvements.

- **Sign SGD** — Replace gradient with its sign: `W -= lr * sign(grad)`. Normalizes gradient magnitudes, helping detect sparse features. Solves k=5 2x faster than standard SGD (7 vs 14 epochs). Surprise: standard SGD also solves k=5 with enough data (n_train=5000). The exp_d "failure" at 61.5% was insufficient data, not algorithm limits.
- **Curriculum learning** — Train on easy configs first, then expand the network. n=10/k=3 (instant) → expand W1 with zero-padded columns → n=30/k=3 (1 epoch) → n=50/k=3 (1 epoch). Result: 14.6x speedup, cracks n=50 which direct training can't. The learned feature detector transfers perfectly because new columns start near zero.
- **Cache-aware MemTracker** — Built CacheTracker with LRU cache simulation to fix the broken ARD metric. Finding: L2 cache (256KB) eliminates ALL misses for both methods. Batch wins on total traffic (13% fewer floats, 16x fewer writes), not cache locality. Single-sample is actually more L1-friendly than batch.
- **Weight decay sweep** — Swept WD across [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]. WD=0.01 already optimal. Only [0.01, 0.05] achieves 100% success rate. The effective LR*WD must be in [0.001, 0.005] — extremely narrow.
- **Per-layer + batch** — Combined per-layer forward-backward with mini-batch SGD. Works (converges) but not valuable: single-sample SGD is 8x faster in epochs, and the per-layer re-forward pass adds 3.7x wall-time overhead.

---

## [0.6.0] - 2026-03-04

### Research experiments (Round 1 — measuring energy on the working solution)

With 20-bit solved, round 1 asked: how energy-efficient is our solution, and can we improve it? We measured ARD across training variants, tested alternative algorithms, and mapped the scaling frontier.

- **Exp A: ARD on winning config** — Measured energy proxy (ARD) on the config that actually works. Per-layer forward-backward gives 3.8% improvement (17,299 vs 17,976 floats). But W1 (the big weight matrix) dominates at 75% of all float reads — this fundamentally caps what operation reordering can achieve at ~10%.
- **Exp B: Batch ARD** — Investigated whether mini-batch SGD is more energy-efficient. Surprising result: batch-32 shows 17x higher ARD in our metric, but does 16x fewer parameter writes. The ARD metric doesn't model cache — on real hardware where W1 fits in L2, batch would win. This exposed a fundamental limitation of our measurement tool.
- **Exp C: Per-layer on 20-bit** — Tested the "fundamentally different algorithm" from Sprint 1's conclusion: update each layer before proceeding to the next. Result: converges identically to standard backprop (99.5%, same epoch count). Free 3.8% ARD improvement with zero accuracy cost. Safe to use everywhere.
- **Exp D: Scaling frontier** — Mapped where standard SGD breaks. k=3 works to n~30-45. n=50/k=3 fails (54%). k=5 is categorically impractical at any n (~200,000 epochs for n=20). The boundary is at n^k > 100,000 iterations, matching the theoretical SQ lower bound.
- **Exp E: Forward-Forward** — Tested Hinton's local learning algorithm (no backward pass). Solves 3-bit perfectly but fails 20-bit (58.5%). Has **25x WORSE ARD** than backprop — opposite of hypothesis. The "local learning" advantage requires 10+ layer networks where backprop stores many activations; our 2-layer MLP is too small to benefit.
- **Exp F: Prompting strategies** — Documented the meta-workflow that solved the problem: literature search → compare against published baselines → diagnose the gap → fix → verify. The single most effective prompt was asking Claude to compare our hyperparams against Barak et al. 2022.

### Speed
- **`fast.py`**: numpy-accelerated training solves 20-bit in 0.12s average (220x faster than pure Python). hidden=200, n_train=1000, batch=32.

### Infrastructure
- LAB.md — protocol for autonomous Claude Code experiment sessions
- DISCOVERIES.md — accumulated knowledge base from all experiments
- `_template.py` — copy-and-modify experiment starter

---

## [0.5.0] - 2026-03-04

### Solving 20-bit sparse parity (the breakthrough)

The pipeline from v0.4.0 got 54% accuracy on 20-bit — coin flip. We did a literature review (6 papers), diagnosed the gap, and fixed it.

**The problem was hyperparameters, not the algorithm.** Our LR=0.5 was 5x too high (literature uses 0.1), we used single-sample SGD instead of mini-batch (batch=32), and had too little training data (200 vs 500+ samples). One arxiv search fixed all of it.

- **Exp 1: Fix Hyperparams** — Changed LR 0.5→0.1, batch_size 1→32, n_train 200→500. Result: 99% accuracy with classic grokking — stuck at 50% for 40 epochs, then phase transition to 99% in ~10 epochs. The hidden progress metric ||w_t - w_0||_1 grew steadily throughout, confirming Barak et al. 2022's theory.
- **Exp 4: GrokFast** — Tested the EMA gradient filter from Lee et al. 2024 (amplify slow gradient components to accelerate grokking). Result: counterproductive. With correct hyperparams, baseline SGD hits 100% in 5 epochs (22.7s). GrokFast took 12 epochs and never reached 100%. Lesson: don't apply tricks designed for one regime (extended memorization) to another (fast convergence).
- Literature review: 6 key papers (Barak 2022, Kou 2024, Merrill 2023, GrokFast, NTK grokking, SLT phase transitions)
- Research plan for autonomous experiment cycle
- Results organized into per-run directories with auto-generated index

---

## [0.4.0] - 2026-03-03

### Added
- Complete sparse parity pipeline (all 5 phases)
    - 3 training variants: standard backprop, fused, per-layer
    - MemTracker for ARD measurement
    - JSON + markdown + plot output
- 20/20 tests passing
- Per-run results directories with index lookup

---

## [0.3.0] - 2026-03-03

### Added
- MkDocs site with Material theme
- Mermaid diagrams in context and findings
- Changelog tracking

---

## [0.2.0] - 2026-03-03

### Added
- `src/sync_google_docs.py` -- standalone script to sync Google Docs to markdown
- Auto-extracted references (`docs/references_auto.md`)
- Expanded homework archive covering all 7 meetings

---

## [0.1.0] - 2026-03-03

### Added
- Initial research environment setup
- Converted 3 Google Docs to local markdown:
    - Challenge #1: Sparse Parity (spec)
    - Sutro Group Main (meeting index)
    - Yaroslav Technical Sprint 1 (sprint log)
- Extracted 30+ hyperlinks into `docs/references.md`
- Directory structure: docs, findings, plans, research, src
- `CLAUDE.md` with project context for AI assistants
- `CONTEXT.md` with project background and timeline
- Sprint 1 findings documented
- Sprint 2 plan drafted
- Meeting notes index with all external links
- Lectures and homework directories
- `.gitignore` for Python/Jupyter/IDE files

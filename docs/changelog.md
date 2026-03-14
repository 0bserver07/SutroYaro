# Changelog

All notable changes to this research workspace.

## [0.19.0] - 2026-03-14

### GPU energy measurement via Modal Labs

- Added `bin/gpu_energy.py`: runs harness methods on NVIDIA L4 via Modal Labs, measures actual watts via pynvml
- First real energy results: SGD uses 18.7J, KM uses 144mJ (130x gap). ARD tracks real joules within an order of magnitude.
- GPU idles at ~12.5W. Methods under 5ms finish before the power sampler can take a reading.
- Findings: `findings/gpu_energy_baseline.md` with proxy-vs-real comparison
- Added scripts table to tooling overview page with examples for all `bin/` scripts
- Updated README project structure with `bin/reproduce-all` and `bin/gpu_energy.py`
- Cost: under $0.01 per run

---

## [0.18.0] - 2026-03-14

### Reproduce-all script

- Added `bin/reproduce-all`: runs all 14 experiments across 3 challenges, verifies results match baselines
- Supports `--budget MS` flag to skip experiments over a time budget (Yaroslav's "Spark 7 constraint": only run what fits in 1980s compute budgets)
- All 14 experiments reproduce in 0.28 seconds. With `--budget 10` (10ms), 6 pass, 8 skip, 0.08 seconds total
- GF(2) on sum/and marked SKIP (expected fail, not a regression)

---

## [0.17.0] - 2026-03-14

### Three challenges, adding-a-challenge guide, Antigravity validation

- Added sparse sum (Challenge 2) and sparse AND (Challenge 3) to the harness
- Created `docs/research/adding-a-challenge.md`: step-by-step guide for agents and contributors to add new tasks
- Sparse AND was added by Google Antigravity (agent IDE) following the guide without human help, validating that the guide works for autonomous agents
- Harness now supports `--challenge` flag: `sparse-parity` (default), `sparse-sum`, `sparse-and`
- Sparse sum baselines: SGD 100% in 1 epoch (ARD 20), KM 100% (ARD 92)
- Sparse AND baselines: SGD 100% in 4 epochs, KM needs 20 samples (not 5) due to 1/2^(k-1) influence signal
- Updated search_space.yaml, questions.yaml, DISCOVERIES.md, TODO.md, baseline_check.py for both challenges

---

## [0.16.0] - 2026-03-14

### Repo migration, multi-topic Telegram sync, and skills

- Moved repo from `0bserver07/SutroYaro` to `cybertronai/SutroYaro` (updated 15 files, git remote, GitHub Pages URL)
- Telegram sync now pulls 6 topics in priority order (chat-yad, chat-yaroslav, challenge #1, General, In-person, Introductions) to separate JSON files
- Added `sutro-sync` skill: session-start routine for Telegram, Google Docs, GitHub checks
- Added `sutro-context` skill: research context loader (DISCOVERIES.md, open questions, recent discussion)
- Re-synced all 15 Google Docs with upstream changes
- Both of Andy's PRs merged (#2 TODO cleanup, #3 GF(2) noise experiment), Issue #1 closed
- Deploy workflow confirmed working on new org

---

## [0.15.0] - 2026-03-11

### Peer research protocol and autonomous agent infrastructure

Inspired by analysis of Karpathy's autoresearch and trevin-creator's Tiny-Lab, but built for multi-researcher use with our own naming conventions.

**New files:**
- `AGENT.md` -- machine-executable experiment loop for autonomous sessions
- `src/harness.py` -- locked evaluation harness (GF2, SGD, KM, Fourier, SMT) with CLI
- `research/search_space.yaml` -- bounded mutation space (16 methods, allowed parameter values)
- `research/questions.yaml` -- dependency graph of 12 research questions (9 resolved, 6 open)
- `research/log.jsonl` -- all 33 experiments from DISCOVERIES.md in machine-readable format
- `results/scoreboard.tsv` -- auto-generated leaderboard from log.jsonl
- `results/progress.png` -- ARD progress chart over experiment history
- `checks/env_check.py` -- pre-flight environment verification
- `checks/baseline_check.py` -- re-establish GF2/SGD/KM baselines per machine
- `bin/run-agent` -- tool-agnostic launcher with looped mode, circuit breaker, PID lock
- `bin/merge-findings` -- merge contributor log.jsonl entries via PR
- `bin/analyze-log` -- progress report and chart generation
- `docs/research/peer-research-protocol.md` -- full design doc with nanoGPT migration proposal

**Design choices:**
- Tool-agnostic: `bin/run-agent --tool claude|gemini|custom` works with any AI CLI
- Looped mode: multiple short cycles with fresh context, resilient to crashes
- Circuit breaker: halts if 5+ INVALID in last 20 experiments
- Harness integrity: SHA256 verified before and after each run
- All state in files: any tool that reads/writes files can participate
- Challenge-agnostic log schema: `challenge` field supports sparse parity now, nanoGPT later
- Researcher attribution: `researcher` field in log entries for peer merge

**Updated files:**
- `CLAUDE.md` -- added autonomous research infrastructure section, harness to isolation rule
- `results/scoreboard.tsv` -- generated from full 33-experiment log

---

## [0.14.0] - 2026-03-11

### Feedback tasks from Meeting #8

- Added Data Movement Complexity (DMC) metric to MemTracker (Ding et al., arXiv:2312.14441). DMC = sum of sqrt(stack_distance) for all float accesses. Baseline: ARD 4,104 / DMC 300,298.
- Confirmed stack distance already implemented (tracker clock advances by buffer size, not instruction count)
- Added metric isolation rule to LAB.md (rule #9): agents cannot modify measurement code
- Created task tracker in `docs/tasks/` with 6 tasks from Meeting #8 feedback
- Updated baselines table in LAB.md with DMC column
- Reproduced Germain's hidden=64 result: ARD drops 68% but ARD/float is identical (0.36). Smaller model, not better locality.
- Reviewed linear classifier paper (arXiv:2309.06979): CoT-based, not applicable to one-shot sparse parity benchmark

---

## [0.13.0] - 2026-03-10

### Meeting #8 docs and sync runbook

- Synced 6 new Google Docs from Meeting #8 (09 Mar 2026): notes, AI notes, Yaroslav knowledge sprint 2, Yaroslav GF(2) verification, Michael's Claude approach, The Bigger Picture roadmap
- Added all new docs to MkDocs nav with cross-reference headers
- Updated meetings index and notes pages with Meeting #8 summary
- Created sync runbook (`docs/tooling/sync-runbook.md`) with weekly/daily/per-session checklists
- Added GitHub PR/issue checking to CLAUDE.md "Before Pushing" checklist
- Total synced Google Docs: 15 (up from 9)

---

## [0.12.0] - 2026-03-09

### GF(2) noise tolerance experiment

- Added exp_gf2_noise experiment testing algebraic solver with label noise
- Key finding: Basic GF(2) fails at 1% noise; robust subset-sampling recovers up to 10-15%
- New experiment: `src/sparse_parity/experiments/exp_gf2_noise.py`
- Findings: `findings/exp_gf2_noise.md`
- Updated DISCOVERIES.md with noise tolerance results

---

## [0.11.0] - 2026-03-07

### Homepage and documentation refresh

- Rewrote homepage as a proper introduction (was jumping straight to "SOLVED")
- Updated context page with Phase 2 findings and timeline
- Synced 8 Google Docs with upstream changes (new Bookmarks section, Yaroslav link)
- Added "How to find things in Sutro Group" doc to sync config and nav
- Fixed sync script to preserve cross-reference headers across re-syncs

---

## [0.10.0] - 2026-03-07

### Review fixes and project documentation

- Fixed RL bandit tracker bug (stale loop variable `i` instead of `arm_idx`)
- Re-sorted TL;DR table by verdict tier then speed, ranked Target Propagation as #33
- Updated CLAUDE.md with Phase 2 results, best methods table, Telegram sync reference
- Added AGENTS.md documenting how 17 parallel Claude Code agents were used
- Added `.env.example` for Telegram API credentials
- Gitignored `messages.json` (contains real group messages)

---

## [0.9.0] - 2026-03-07

### Phase 2: 17 experiments + Practitioner's Field Guide

**Phase 2 experiments** dispatched 17 independent Claude Code agents in parallel, each testing a different algorithmic approach:

- **Algebraic/Exact**: GF(2) Gaussian elimination (509 us, 240x faster than SGD), Kushilevitz-Mansour influence estimation (ARD 1,585, 724x better than Fourier), SMT backtracking
- **Information-Theoretic**: Mutual Information, LASSO, MDL Compression, Random Projections -- all solve it, none beats Fourier meaningfully
- **Local Learning Rules**: Hebbian, Predictive Coding, Equilibrium Propagation, Target Propagation -- all failed at chance level (parity requires k-th order interaction detection)
- **Hardware-Aware**: Tiled W1 (software ARD worsened), Pebble Game (2.2% energy savings), Binary Weights (fails at n=20)
- **Alternative Framings**: Genetic Programming (exact formula but doesn't scale), RL Bit Querying (ARD of 1 at inference), Decision Trees (greedy splitting can't find secret bits)

**Practitioner's Field Guide** (`docs/research/survey.md`): 4,500-word survey ranking all 33 experiments with decision framework, 10 generalized principles, and full AI research methodology.

**Telegram sync tooling**: `sync_telegram.ts` pulls messages from Sutro Group topic threads via MTProto. Full setup guide in docs/tooling/automation.md.

Updated DISCOVERIES.md, mkdocs nav (survey + 17 findings pages), and research index.

---

## [0.8.0] - 2026-03-04

### Blank-slate approaches (Round 3, no neural nets, no SGD)

Rounds 1-2 were incremental variations on the same MLP+SGD recipe. Round 3 reframes the problem: sparse parity as a **search problem**, not a learning problem.

For k=3 parity with secret indices {a,b,c}, the label is `x[a] * x[b] * x[c]`. You don't need a neural network. You need to search over C(n,k) possible subsets and test which one matches the data.

- **Fourier / Walsh-Hadamard solver**: Sparse parity IS a Fourier coefficient. For each candidate k-subset S, compute `mean(y * product(x[:, S]))`. The true secret gives correlation ~1.0, everything else gives ~0. No training, no gradients, no iterations. Result: **13x faster than SGD** (0.009s vs 0.12s for n=20/k=3), solves n=50/k=3 and n=20/k=5 trivially where SGD struggles. Only needs 20 samples. Scales to k=7 before combinatorial explosion (C(n,k) subsets).

- **Evolutionary / random search**: Randomly sample k-subsets, test if `product(x[:, subset])` matches all labels. For n=20/k=3 it takes ~881 random tries (0.011s). Evolutionary search with mutation+crossover solves it in fewer evaluations but more wall time. Solves n=50/k=3 in 0.14s, a config SGD fails on entirely.

- **Feature selection**: Tried to decompose into "find the bits" then "classify." Exhaustive combo search works (178-1203x fewer ops than SGD). But the clever approaches fail: pairwise correlations are provably zero for parity (E[y * x_i * x_j] = 0 for ALL pairs, even correct ones). Greedy forward selection also fails. Parity has **zero low-order statistical signatures**. You must test the full k-way interaction. Neural nets need so many iterations because they implicitly search the combinatorial space via gradient descent.

**When to use what**:
- k ≤ 7: Fourier/random search wins (milliseconds, exact, guaranteed)
- k ≥ 10: C(n,k) explodes, SGD's implicit search via gradients becomes the only feasible path
- k = 8-9: hybrid approaches may work (combinatorial search with pruning)

---

## [0.7.0] - 2026-03-04

### Research experiments (Round 2, autonomous agent team)

Round 2 explored variations on the working SGD solution: different optimizers, hyperparameter sweeps, and energy measurement improvements.

- **Sign SGD**: Replace gradient with its sign: `W -= lr * sign(grad)`. Normalizes gradient magnitudes, helping detect sparse features. Solves k=5 2x faster than standard SGD (7 vs 14 epochs). Standard SGD also solves k=5 with enough data (n_train=5000). The exp_d "failure" at 61.5% was insufficient data, not algorithm limits.
- **Curriculum learning**: Train on easy configs first, then expand the network. n=10/k=3 (instant) → expand W1 with zero-padded columns → n=30/k=3 (1 epoch) → n=50/k=3 (1 epoch). Result: 14.6x speedup, cracks n=50 which direct training can't. The learned feature detector transfers because new columns start near zero.
- **Cache-aware MemTracker**: Built CacheTracker with LRU cache simulation to fix the broken ARD metric. Finding: L2 cache (256KB) eliminates ALL misses for both methods. Batch wins on total traffic (13% fewer floats, 16x fewer writes), not cache locality. Single-sample is more L1-friendly than batch.
- **Weight decay sweep**: Swept WD across [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]. WD=0.01 already optimal. Only [0.01, 0.05] achieves 100% success rate. The effective LR*WD must be in [0.001, 0.005], an extremely narrow range.
- **Per-layer + batch**: Combined per-layer forward-backward with mini-batch SGD. Converges but not useful: single-sample SGD is 8x faster in epochs, and the per-layer re-forward pass adds 3.7x wall-time overhead.

---

## [0.6.0] - 2026-03-04

### Research experiments (Round 1, measuring energy on the working solution)

With 20-bit solved, round 1 asked: how energy-efficient is our solution, and can we improve it?

- **Exp A: ARD on winning config**: Measured energy proxy (ARD) on the working config. Per-layer forward-backward gives 3.8% improvement (17,299 vs 17,976 floats). W1 (the big weight matrix) dominates at 75% of all float reads, capping operation-reordering improvements at ~10%.
- **Exp B: Batch ARD**: Batch-32 shows 17x higher ARD in our metric, but does 16x fewer parameter writes. The ARD metric doesn't model cache. On real hardware where W1 fits in L2, batch would win. This exposed a limitation of our measurement tool.
- **Exp C: Per-layer on 20-bit**: Update each layer before proceeding to the next. Converges identically to standard backprop (99.5%, same epoch count). Free 3.8% ARD improvement with zero accuracy cost.
- **Exp D: Scaling frontier**: Mapped where standard SGD breaks. k=3 works to n~30-45. n=50/k=3 fails (54%). k=5 is impractical at any n (~200,000 epochs for n=20). The boundary is at n^k > 100,000 iterations, matching the theoretical SQ lower bound.
- **Exp E: Forward-Forward**: Hinton's local learning algorithm (no backward pass). Solves 3-bit but fails 20-bit (58.5%). Has **25x WORSE ARD** than backprop, opposite of hypothesis. The locality advantage requires 10+ layer networks; our 2-layer MLP is too small to benefit.
- **Exp F: Prompting strategies**: Documented the meta-workflow: literature search → compare against published baselines → diagnose the gap → fix → verify. The most effective prompt was asking Claude to compare our hyperparams against Barak et al. 2022.

### Speed
- **`fast.py`**: numpy-accelerated training solves 20-bit in 0.12s average (220x faster than pure Python). hidden=200, n_train=1000, batch=32.

### Infrastructure
- LAB.md: protocol for autonomous Claude Code experiment sessions
- DISCOVERIES.md: accumulated knowledge base from all experiments
- `_template.py`: copy-and-modify experiment starter

---

## [0.5.0] - 2026-03-04

### Solving 20-bit sparse parity

The pipeline from v0.4.0 got 54% accuracy on 20-bit, a coin flip. Literature review of 6 papers diagnosed the gap.

**The problem was hyperparameters, not the algorithm.** Our LR=0.5 was 5x too high (literature uses 0.1), we used single-sample SGD instead of mini-batch (batch=32), and had too little training data (200 vs 500+ samples). One arxiv search fixed all of it.

- **Exp 1: Fix Hyperparams**: Changed LR 0.5→0.1, batch_size 1→32, n_train 200→500. Result: 99% accuracy with classic grokking, stuck at 50% for 40 epochs, then phase transition to 99% in ~10 epochs. The hidden progress metric ||w_t - w_0||_1 grew steadily throughout, confirming Barak et al. 2022's theory.
- **Exp 4: GrokFast**: Tested the EMA gradient filter from Lee et al. 2024 (amplify slow gradient components to accelerate grokking). Counterproductive. With correct hyperparams, baseline SGD hits 100% in 5 epochs (22.7s). GrokFast took 12 epochs and never reached 100%. Lesson: don't apply tricks designed for one regime (extended memorization) to another (fast convergence).
- Literature review: 6 papers (Barak 2022, Kou 2024, Merrill 2023, GrokFast, NTK grokking, SLT phase transitions)
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
- `src/sync_google_docs.py`: standalone script to sync Google Docs to markdown
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

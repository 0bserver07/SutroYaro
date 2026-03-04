# Homework

## Sparse Parity Challenge — COMPLETED

**Drosophila of Learning** - all 5 original tasks solved.

Full spec: [challenge-1-sparse-parity.md](../google-docs/challenge-1-sparse-parity.md)

### Original Tasks (all done)

- [x] Generate training/testing datasets using random positive/negative numbers for XOR/parity
- [x] Build a neural net that solves the task (>90% accuracy) — **100% achieved**
- [x] Estimate energy via Average Reuse Distance (ARD) — **MemTracker + CacheTracker built**
- [x] Prompt AI to improve the algorithm's ARD — **per-layer gives 3.8%, fused gives 1.3%**
- [x] Scale to 3-bit parity with 17 noise "dirty" bits (20 total) — **solved in 0.12s**

### Beyond the original homework (16 experiments total)

- Solved n=50/k=3 via curriculum learning (14.6x speedup)
- Solved k=5 with Sign SGD and standard SGD (more training data)
- Tested Forward-Forward (25x worse ARD — not viable)
- Built cache-aware MemTracker with LRU simulation
- Discovered blank-slate approaches: Fourier solver 13x faster than SGD for small k
- Documented prompting strategies for AI-assisted research

See [Changelog](../changelog.md) for full version history and [DISCOVERIES.md](https://github.com/0bserver07/SutroYaro/blob/main/DISCOVERIES.md) for accumulated knowledge.

### Tips from the group
- Keep iteration time <2 seconds — use `fast.py` (numpy, 0.12s per solve)
- Change one thing at a time: correctness, then speed, then energy
- Priority order: correctness > wall-clock time > energy usage
- Compare against published baselines FIRST (this alone solved 20-bit)

### Reference implementation
- Fast solver: `src/sparse_parity/fast.py` (numpy, 0.12s)
- Full pipeline: `src/sparse_parity/run.py` (3 variants, ARD, plots)
- [cybertronai/sutro repo](https://github.com/cybertronai/sutro) (Yaroslav's original)

---

## Past Homework Archive

### Meeting #5 (16 Feb 26) - Karpathy Names Task

**Status**: Not yet attempted in this repo.

**Goal**: Optimize a character-level model for energy efficiency. Take 1000 random names from Karpathy's [makemore/names.txt](https://github.com/karpathy/makemore/blob/master/names.txt), predict last 3 characters.

**Emmett's approach**: Pure-Python GPT, reduced memory from 80MB to 35MB. [Full implementation](https://docs.google.com/document/d/1DAwx_gohi6tomMPkb_fETAIuxIyHgLtC5OPD_qpGpqg/edit?tab=t.0)

### Meeting #2 (26 Jan 26) - Forward-Forward Algorithm

**Status**: Implemented and tested in [Exp E](../findings/exp_e_forward_forward.md). FF solves 3-bit but fails 20-bit, 25x worse ARD.

### Meeting #3 (02 Feb 26) - Joules Measuring

**Status**: MemTracker and CacheTracker built. See [Exp B](../findings/exp_b_batch_ard.md) and [Exp Cache](../findings/exp_cache_ard.md).

### Meeting #1 (19 Jan 26) - Energy-Efficient Training Intro

**Notes**: [sutro meeting #1](https://docs.google.com/document/d/1ZsH26hVvbZBOshwA1KgdX5AK5zw9W0CzqZuXLa5fIlo/edit?tab=t.0)

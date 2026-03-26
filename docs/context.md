# Context

## What is this?

A research environment for the Sutro Group's work on energy-efficient AI training. The group's thesis: go back to 1960s-era AI problems and reinvent learning algorithms using modern tools (AI agents, compute), with energy efficiency as the optimization target.

## Why Sparse Parity?

Sparse parity is the "drosophila" of learning tasks:

- **Simplest non-trivial** learning problem (XOR was the example Minsky used to trigger the AI winter)
- **Easy to scale** difficulty (add noise bits)
- **Fast to iterate** (0.12s with numpy, <2s even in pure Python)
- **Exposes** memory access patterns in backprop
- **Well-studied** in theory (Barak et al. 2022, Kou et al. 2024)

## What We Found

36 experiments across three phases. Full ranked results in the [Practitioner's Field Guide](research/survey.md).

### Phase 1: SGD optimization (16 experiments)

The 20-bit problem was "unsolvable" at LR=0.5. At LR=0.1 it solves in 5 epochs. From there we optimized ARD within the SGD framework, hitting a ceiling at ~10% because W1 accounts for 75% of all float reads. Forward-Forward has 25x worse ARD than backprop for 2-layer networks. Curriculum learning broke the scaling wall (n=50). The cache simulator showed L2 eliminates all misses.

### Phase 2: Broad search (17 experiments)

Parity is linear over GF(2). GF(2) Gaussian elimination solves in 509 microseconds, 240x faster than SGD. Kushilevitz-Mansour influence estimation achieves ARD 1,585 (724x better than Fourier). All four local learning rules (Hebbian, Predictive Coding, Equilibrium Propagation, Target Propagation) fail at chance level because parity requires k-th order interaction detection.

### Phase 3: GrokFast (3 experiments, by Seth Stafford)

GrokFast v2 (EMA gradient filter) gives 2.5x speedup on k=5 but fails on large n. Compounding GrokFast with curriculum learning solves n=50/k=5 in 14 epochs. The scaling frontier: n=200/k=3 in 95ms, but k=5 hits a wall at n=100.

### Key insight

For small k, sparse parity is a search problem, not a learning problem. The neural network was solving an easy problem the hard way.

## The Bigger Picture

Yaroslav's [roadmap](google-docs/bigger-picture.md) defines three axes of progress:

1. **Process** (orange): improve how agents find better algorithms. Multiple members built independent harnesses (Claude Code, Replit Research OS, plain Claude). The process itself is the product.
2. **Metric** (green): make the energy proxy more realistic. Started with ARD, added DMC (Data Movement Complexity, Ding et al.). Next step: actual GPU measurement on an H100.
3. **Problem** (blue): make the task harder. Sparse parity is practice. The final exam is energy-efficient training of [nanoGPT](https://github.com/karpathy/nanoGPT).

Take small steps along one axis at a time to keep complexity manageable. The group explicitly avoids premature partitioning (optimizing training but not inference, or math but not kernels).

## Timeline

```mermaid
gantt
    title Sutro Group Timeline
    dateFormat YYYY-MM-DD
    section Meetings
    Meeting 1 - Energy Intro          :m1, 2026-01-19, 1d
    Meeting 2 - Forward-Forward       :m2, 2026-01-26, 1d
    Meeting 3 - Joules Measuring      :m3, 2026-02-02, 1d
    Meeting 4 - Beauty to Joules      :m4, 2026-02-09, 1d
    Meeting 5 - Intelligence/Joule    :m5, 2026-02-16, 1d
    Meeting 6 - Presentations         :m6, 2026-02-23, 1d
    Meeting 7 - Sparse Parity         :m7, 2026-03-02, 1d
    Meeting 8 - Demos + Roadmap       :m8, 2026-03-09, 1d
    Meeting 9 - Roadmap + GF(2)       :m9, 2026-03-16, 1d
    Meeting 10 - DMC Results          :m10, 2026-03-23, 1d
    section Research
    Sprint 1 (ARD baseline)           :s1, 2026-03-02, 1d
    Sprint 2 (solve 20-bit)           :s2, 2026-03-03, 2d
    Phase 1 (16 experiments)          :p1, 2026-03-04, 2d
    Phase 2 (17 parallel agents)      :p2, 2026-03-06, 1d
    Survey written                    :sv, 2026-03-06, 1d
    DMC metric + task triage          :dm, 2026-03-11, 1d
    DMC baseline sweep                :db, 2026-03-22, 1d
    Seth's GrokFast PRs (3)           :gf, 2026-03-23, 1d
    RL eval environment               :ev, 2026-03-24, 1d
    Agent infrastructure              :ai, 2026-03-24, 1d
```

## People

| Name | Role / Focus |
|------|-------------|
| **Yad** | Created this repo (SutroYaro), built the Claude Code autonomous research lab: parallel agent teams, experiment templates, DISCOVERIES.md knowledge accumulation |
| **Yaroslav** | Sutro Group founder, technical sprints, algorithm work, [cybertronai/sutro](https://github.com/cybertronai/sutro) |
| **Emmett** | Aster agentic loop framework, 2x energy improvement on microgpt |
| **G B** | Architecture experiments (depth-1/hidden-64, ARD ~33-35) |
| **Germaine** | Presentations, implementations |
| **Andy Zhang** | ML consultant, GitHub contributor ([zh4ngx](https://github.com/zh4ngx)), GF(2) noise experiment, TODO cleanup |
| **Michael Keating** | Former energy tech CEO (Scoot), Claude-based sparse parity approach |
| **Seth Stafford** | GrokFast experiments (3 merged PRs: v2, curriculum compound, scaling frontier), healthcare AI |
| **Barak** | Modal workflow |
| **Jamie Simon** | Forward-Forward implementation |
| **Jonathan Belay** | Deterministic methods, spectral graph theory |
| **Anish Tondwalkar** | Former Google hardware engineer (inference chips), RL environments startup |
| **Uliana Popov** | Applied AI, temperature tuning suggestions |
| **Josh (Joshua Marks)** | Hardware engineer, SRAM/DRAM properties, circuit diagrams |
| **Jack Schenkman** | Research scientist, EE background, ASIC design |
| **Preston Schmittou** | 500-parameter transformers, message passing research |
| **Caleb Sirak** | DIY AI supercomputer ("Howard") |

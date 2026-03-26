# Sparse Parity Research Lab

> This file is the entry point for any Claude Code session running experiments.
> Read this FIRST before doing anything.

## How This Lab Works

This is an autonomous research lab. Each experiment follows a strict template so findings accumulate and future sessions can build on past work without re-reading everything.

### Quick Start for New Experiment

```bash
# 1. Read this file (LAB.md)
# 2. Read DISCOVERIES.md for what's known so far
# 3. Pick an open question from DISCOVERIES.md or TODO.md
# 4. Create experiment using the template below
# 5. Run, record, commit
```

### Directory Layout

```
LAB.md                  # You are here — lab protocol
DISCOVERIES.md          # Accumulated knowledge (READ THIS)
TODO.md                 # Open research tasks

src/sparse_parity/
  experiments/
    _template.py        # Copy this to start a new experiment
    exp1_*.py           # Completed experiments
    exp_a_*.py
    ...

findings/
  {exp_name}.md         # One file per experiment, strict format

results/
  {exp_name}/
    results.json        # Machine-readable metrics
    *.png               # Plots (optional)
```

### The Experiment Lifecycle

```
1. HYPOTHESIS  →  What do you expect and why?
2. SETUP       →  Config, code, what you're measuring
3. RUN         →  Execute, capture all output
4. RESULTS     →  Numbers in a table
5. ANALYSIS    →  Why did it work/fail? What's surprising?
6. NEXT        →  What should be tried next based on this?
7. COMMIT      →  findings/ + results/ + experiments/
```

---

## Experiment Template (findings/{exp_name}.md)

Every finding MUST follow this format exactly:

```markdown
# Experiment {ID}: {Title}

**Date**: YYYY-MM-DD
**Status**: SUCCESS | PARTIAL | FAILED
**Answers**: {Which question from DISCOVERIES.md does this address?}

## Hypothesis

{One sentence: "If we do X, then Y will happen because Z."}

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | |
| k_sparse | |
| hidden | |
| lr | |
| wd | |
| batch_size | |
| max_epochs | |
| n_train | |
| seed | |
| method | {standard/fused/perlayer/forward-forward/sign-sgd/...} |

## Results

| Metric | Value |
|--------|-------|
| Best test accuracy | |
| Epochs to >90% | |
| Wall time | |
| Weighted ARD | |
| ARD improvement vs baseline | |

## Key Table

{The ONE comparison table that tells the story}

## Analysis

### What worked
{Bullet points}

### What didn't work
{Bullet points}

### Surprise
{The one thing you didn't expect}

## Open Questions (for next experiment)

- {Question 1 — specific enough to be an experiment}
- {Question 2}

## Files

- Experiment: `src/sparse_parity/experiments/{exp_name}.py`
- Results: `results/{exp_name}/results.json`
```

---

## Code Template (src/sparse_parity/experiments/_template.py)

See `src/sparse_parity/experiments/_template.py` for the code template.

---

## Rules for Autonomous Sessions

1. **Read DISCOVERIES.md first** — don't repeat what's known
2. **One hypothesis per experiment** — change one thing at a time
3. **Always compare against a baseline** — never report absolute numbers alone
4. **Record failures** — a failed hypothesis is still a finding
5. **Update DISCOVERIES.md** — add your finding to the knowledge base
6. **Keep runtime < 5 minutes** — reduce hidden/epochs if needed
7. **Commit locally, don't push** — the human decides when to push
8. **Leave a "Next" section** — so the next session knows what to try
9. **Metric isolation** — never modify measurement code (tracker.py, cache_tracker.py, data.py, config.py, harness.py). Agents that rewrite evaluation code to get better scores are gaming the metric, not improving the algorithm.
10. **Two-phase results** — separate evidence from interpretation:
    - **Phase 1 (evidence)**: experiment code produces `results/{exp_id}/results.json` with raw numbers, config, and environment info. No interpretation in this file.
    - **Phase 2 (findings)**: write `docs/findings/{exp_id}.md` referencing the results JSON. Add hypothesis, analysis, and impact on DISCOVERIES.md here.
    - Never write conclusions in Phase 1. Never put raw data in Phase 2 without linking to Phase 1.
11. **Reproducibility** — see `.claude/rules/experiment-reproducibility.md`. Every experiment must record seed, config, environment, and git commit hash.

## Current Baselines

| Config | Method | Accuracy | ARD | DMC | Time | Reference |
|--------|--------|----------|-----|-----|------|-----------|
| n=20, k=3 | numpy SGD (fast.py) | 100% | — | — | 0.12s | fast.py |
| n=20, k=3 | standard (LR=0.1, batch=32) | 100% | 17,976 | — | — | exp_a |
| n=20, k=3 | standard (single sample, tracked) | 100% | 4,104 | 300,298 | 1.78s | baseline |
| n=20, k=3 | perlayer (LR=0.1) | 99.5% | 17,299 | — | exp_c |
| n=20, k=3 | forward-forward | 58.5% | 277,256 | — | exp_e |
| n=20, k=5 | sign SGD (n_train=5000) | >90% | — | — | exp_sign_sgd |
| n=20, k=5 | standard (n_train=5000) | >90% | — | — | exp_sign_sgd |
| n=30, k=3 | standard (LR=0.1, batch=32) | 94.5% | — | — | exp_d |
| n=50, k=3 | curriculum (n=10→30→50) | >90% | — | — | exp_curriculum |
| n=50, k=3 | standard (direct) | 54% (FAIL) | — | — | exp_d |
| n=3, k=3 | standard | 100% | 10,640 | — | run_20260303_200353 |
| n=3, k=3 | perlayer | 100% | 9,674 | — | run_20260303_200353 |
| n=20, k=3 | fourier (Walsh-Hadamard) | 100% | 1,147,375 | 0.009s | exp_fourier |
| n=50, k=3 | fourier (Walsh-Hadamard) | 100% | — | 0.16s | exp_fourier |
| n=20, k=5 | fourier (Walsh-Hadamard) | 100% | — | 0.14s | exp_fourier |

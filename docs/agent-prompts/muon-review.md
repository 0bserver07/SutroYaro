# Muon Optimizer Literature Review

You are contributing to the **Sutro Group**, a research lab studying energy-efficient AI training. Your task is a literature review of the **Muon optimizer**.

## Project context

Read these files first:
- `DISCOVERIES.md` — what's already known
- `LAB.md` — lab rules (important: do NOT modify measurement code, tracker.py, cache_tracker.py, data.py, config.py, harness.py)
- `findings/_template.md` — reference for expected report format

- We optimize for **energy efficiency**, not just accuracy or speed
- Our benchmark: **sparse parity** (n=20 bits, k=3 secret, 17 noise)
- Our metric: **ByteDMD** (https://github.com/cybertronai/ByteDMD) — successor to DMC. Tracks memory at **byte level** (not per-element), using integer arithmetic. Reference implementation has been hardened against escape hatches (agents were bypassing TrackedArray via np.asarray() → Python ints). See also: https://github.com/cybertronai/ByteDMD-examples for test cases.
- **Baseline SGD**: ~0.12s wall-clock, high DMD (exact ByteDMD score unmeasured — you'd need to run the ByteDMD tracer)
- **Best known**: GF(2) Gaussian Elimination (~509us, low byte-level access count), KM-min (~1ms)

## Task: Research the Muon optimizer

### 1. Read the Muon paper

Primary source: https://kellerjordan.github.io/posts/muon/

Key facts to extract:
- The algorithm: Newton-Schulz iteration on SGD momentum updates to orthogonalize gradients
- How it replaces Adam's element-wise adaptation with matrix orthogonalization
- Performance claims (CIFAR-10, NanoGPT, 1.5B parameter models)
- Computational overhead (<1% FLOP overhead claimed)
- Scope: only for 2D hidden-layer weights, paired with AdamW for embeddings/biases

### 2. Analyze relevance to energy efficiency

Study the ByteDMD metric: https://github.com/cybertronai/ByteDMD

Investigate:
- Does Newton-Schulz iteration **reduce byte-level data movement** compared to Adam's moment tracking (first + second moment buffers)?
- What's the **FLOP/memory tradeoff**? Does it reduce byte accesses at the cost of more compute?
- Does orthogonalization change **reuse distance patterns** compared to Adam or SGD?
- Would Muon help on **small networks** (hidden=200, our sparse parity setup) or only large LLMs?

### 3. Write findings

Create `findings/exp_muon_review.md` following this structure:

```markdown
# Muon Optimizer — Literature Review

## Hypothesis
[What we expect Muon to help with (or not) for energy efficiency]

## Key Facts from Paper
- [5-10 bullets on algorithm, results, limitations]

## Relevance to Sutro Group
[Would Muon improve energy efficiency on sparse parity? Why/why not?]

## Comparison to Our Methods
[How Muon relates to our best methods: SGD, GF(2), KM-min]

## Open Questions
[What we'd need to test experimentally to know for sure]

## References
- [Links to paper, code, follow-ups]
```

## Rules

- **DO NOT** modify any experiment/run code (tracker.py, cache_tracker.py, data.py, config.py, harness.py, fast.py, src/)
- **DO NOT** run experiments — this is a literature review only
- Write findings to `findings/exp_muon_review.md`
- Check `DISCOVERIES.md` first to avoid repeating known results
- Do NOT change LAB.md, DISCOVERIES.md, CLAUDE.md, CODEX.md, or any other project configuration

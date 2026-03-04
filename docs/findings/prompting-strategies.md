# Prompting Strategies for AI-Assisted Research

**Date**: 2026-03-04
**Context**: Sutro Group Challenge #1, Question 3 -- "What are the prompting strategies/approaches that are useful here?"

We used Claude Code as an autonomous research agent to go from 54% accuracy (coin-flip) to 100% accuracy on 20-bit sparse parity (k=3). This document captures the workflow, the prompts that worked, the ones that did not, and a reusable template for future research cycles.

---

## The Workflow That Worked

The successful cycle followed a specific five-stage loop:

```
Literature Search --> Diagnose Gap --> Hypothesis --> Experiment --> Measure --> Iterate
```

### Stage 1: Literature Search

**Prompt pattern**: Ask the AI to search for relevant academic work on your exact problem.

Example prompt:
> "Search arxiv for sparse parity learning neural network. Find the key papers on how SGD learns sparse parity, what hyperparameters they use, and what convergence behavior to expect."

This produced a literature review (`research/sparse-parity-literature.md`) covering six papers:
- Barak et al. 2022 (hidden progress in deep learning)
- Kou et al. 2024 (Sign SGD matching SQ lower bound)
- Merrill et al. 2023 (grokking as circuit competition)
- Lee et al. 2024 (GrokFast)
- Feature learning dynamics under grokking (2024)
- Rubin et al. 2026 (grokking as phase transition)

**Why it worked**: The AI retrieved concrete hyperparameters and convergence expectations from published baselines, not just abstract theory.

### Stage 2: Diagnose the Gap

**Prompt pattern**: Ask the AI to compare your current configuration against what the literature says should work.

The literature review included a table comparing our config to published baselines:

| Parameter | Ours (broken) | Literature (working) | Why it matters |
|-----------|--------------|---------------------|----------------|
| LR        | 0.5          | 0.1                 | Too high = overshoot |
| batch_size | 1           | 32                  | Reduces gradient noise |
| max_epochs | 50          | 500+                | Phase transition needs n^O(k) steps |
| n_train   | 200          | 500-1000            | More data helps generalization |

This diagnosis was the single most valuable output of the entire process. It turned a vague "it doesn't work" into a precise checklist of things to fix.

### Stage 3: Hypothesis and Experiment Plan

**Prompt pattern**: Ask the AI to create a ranked experiment plan with clear hypotheses and success criteria.

The research plan (`docs/plans/2026-03-04-beat-20bit-research-plan.md`) laid out six experiments in priority order, each with:
- A specific hypothesis
- Exact parameter changes
- Success criteria (e.g., ">90% test accuracy")
- A protocol for recording and committing results

### Stage 4: Execute and Measure

**Prompt pattern**: Give the AI one experiment at a time with a clear protocol.

Each experiment followed a fixed template:
1. Create `src/sparse_parity/experiments/exp_{N}_{name}.py`
2. Run, save results to `results/`
3. Write findings to `findings/exp_{N}_{name}.md`
4. Commit with a descriptive message
5. If >90% accuracy, stop and document the winning config

Experiment 1 (fix hyperparameters) hit 99% accuracy at epoch 52, solving the problem.

### Stage 5: Iterate on Failures

When Experiment 4 (GrokFast) made things worse, the AI analyzed WHY and documented the mismatch between the technique's intended regime and our actual regime.

---

## Prompts That Were Effective

### 1. "Search arxiv for [exact problem description]"

Asking for literature on the exact task, not a general topic, produced directly applicable papers with usable hyperparameters.

**Good**: "Search arxiv for sparse parity learning neural network"
**Less good**: "How do I train a neural network better?"

### 2. "What are the practical hyperparameters from this paper?"

Asking for concrete numbers from papers, not just theoretical insights, was critical. The Barak et al. paper contained the exact recipe (LR=0.1, batch_size=32, hidden=1000, hinge loss) that solved our problem.

### 3. "Compare our config against published baselines"

This is the highest-leverage prompt in the entire workflow. It turns the literature review into an actionable diagnosis. The output is a diff between "what we have" and "what works," which directly maps to experiment priorities.

### 4. "Create an experiment plan with ranked hypotheses"

Asking for a prioritized plan prevents the temptation to try the most exciting idea first. Experiment 1 (fix hyperparameters) was less glamorous than Experiment 4 (GrokFast) but was the actual solution.

### 5. "Track hidden progress metrics, not just loss/accuracy"

Asking the AI to track ||w_t - w_0||_1 (weight movement norm) revealed that SGD was making progress even when test accuracy was stuck at 50%. This confirmed the "hidden progress" phenomenon and gave confidence to keep training.

---

## What Did NOT Work

### 1. GrokFast Was Counterproductive

The literature review flagged GrokFast as "our most promising practical trick -- could reduce training from 500 epochs to 5-50." In practice:

- GrokFast was 17x slower (383.7s vs 22.7s)
- Caused 83x more weight movement (441K vs 5.3K L1 norm)
- Never reached 100% accuracy (peaked at 99%)
- Baseline SGD reached 100% in 5 epochs without any tricks

**Lesson**: A technique that works brilliantly in one regime (modular arithmetic with thousands of grokking epochs) can be actively harmful in another (sparse parity where correct hyperparameters give fast convergence). The literature review correctly identified GrokFast as promising, but the regime mismatch was only discovered by running the experiment.

### 2. Blindly Following Paper Defaults Without Understanding the Regime

The GrokFast default parameters (alpha=0.98, lambda=2.0) were tuned for tasks where grokking takes thousands of epochs. Applying them to a task that converges in 5 epochs amplified gradients that were already strong enough, causing instability.

### 3. Trying Fancy Algorithms Before Fixing Basics

If we had jumped straight to GrokFast or Sign SGD without first fixing the hyperparameters, we would have wasted significant time. The problem was not that SGD was insufficient -- it was that our LR was 5x too high and our batch size was 1 instead of 32.

---

## Lessons Learned

1. **Always compare your config against published baselines first.** This is the single highest-ROI step. A five-minute literature search and config comparison solved a problem that would have been unsolvable with algorithmic tricks alone.

2. **The simplest fix should be tried before fancy algorithms.** Hyperparameter correction (Experiment 1) solved the problem. GrokFast (Experiment 4) made it worse. Prioritize by simplicity, not novelty.

3. **Track hidden progress metrics, not just loss/accuracy.** Weight movement norm (||w_t - w_0||_1) was a leading indicator of convergence. Without it, we might have given up during the "silent" phase when test accuracy was stuck at 50%.

4. **Literature techniques do not always transfer to your regime.** GrokFast was designed for extended memorization phases (thousands of epochs). Our task converged in 5 epochs. Always check whether the assumptions behind a technique match your setting.

5. **One change at a time, with clear success criteria.** The experiment plan tested each idea independently with a defined success threshold (>90% test accuracy). This made it trivial to identify what worked and what did not.

6. **Document failures as carefully as successes.** The GrokFast finding (Experiment 4) is arguably more informative than the hyperparameter fix (Experiment 1), because it identifies a common research pitfall: importing techniques from a different regime without verifying the assumptions.

---

## Reusable Template for Future Research Cycles

### Phase 1: Diagnose (30 minutes)

```
Prompt: "Search arxiv for [exact task description]. Find the key papers,
         what hyperparameters they use, and what convergence behavior to expect."

Prompt: "Compare our current configuration against published baselines.
         Produce a table showing each parameter, our value, the literature value,
         and why the difference matters."
```

Output: Literature review + gap diagnosis table.

### Phase 2: Plan (15 minutes)

```
Prompt: "Create a ranked experiment plan. Each experiment should have:
         a hypothesis, exact parameter changes, success criteria, and
         a protocol for saving results."
```

Output: Numbered experiment list, ordered by expected impact, simplest first.

### Phase 3: Execute (per experiment)

```
Prompt: "Run Experiment N. Create the script, run it, save results,
         write findings, and commit. If it hits the success criteria, stop."
```

Output: One findings file per experiment, committed to the repo.

### Phase 4: Analyze Failures

```
Prompt: "Experiment N failed / was counterproductive. Analyze WHY.
         Does the technique's intended regime match our actual regime?
         What assumptions were violated?"
```

Output: Updated findings with root-cause analysis.

### Phase 5: Iterate or Ship

If the problem is solved, document the winning configuration and move on.
If not, return to Phase 1 with the new information from failed experiments.

---

## Summary

The core insight is that **AI-assisted research works best when the AI is given a structured workflow**, not open-ended instructions. The sequence "literature search, gap diagnosis, ranked experiments, one-at-a-time execution, failure analysis" produced a solution in under 2 hours. The most effective single prompt was asking the AI to compare our configuration against published baselines -- that one comparison contained the entire fix.

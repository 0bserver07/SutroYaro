\!\!\! info "Cross-references"
    **Source**: [Google Doc](https://docs.google.com/document/d/13uAQfG_ola3vt1hHFo3A8ThUeV-nBVQK/edit) · [Meeting #8 summary](../meetings/notes.md#meeting-8-09-mar-26-demos-and-roadmap)
    **Related**: [Challenge #1](challenge-1-sparse-parity.md) · [Survey](../research/survey.md)


Sutro Group Challenge \#3: Sparse Parity^\[a\](#cmnt1)\[b\](#cmnt2)^

Exploration Results and Next Steps

Opus 4.6 + Michael Keating \| March 2026

Sutro Group Challenge \#3: Sparse Parity^\[a\](#cmnt1)\[b\](#cmnt2)^

Exploration Results and Next Steps

Opus 4.6 + Michael Keating \| March 2026

Sutro Group Challenge \#3: Sparse Parity^\[a\](#cmnt1)\[b\](#cmnt2)^

Exploration Results and Next Steps

Opus 4.6 + Michael Keating \| March 2026

Sutro Group Challenge \#3: Sparse Parity^\[a\](#cmnt1)\[b\](#cmnt2)^

Exploration Results and Next Steps

Opus 4.6 + Michael Keating \| March 2026

# The Problem 

Yaroslav's challenge asks us to solve sparse parity --- identifying 3 secret bits among 20 total inputs and learning their XOR relationship --- as a minimal benchmark for inventing energy-efficient learning algorithms. The task must complete in under 2 seconds, achieve over 90% accuracy, and we must measure memory access patterns (Average Reuse Distance) as a proxy for energy cost. This is the \"drosophila\" of learning: simple enough to iterate on rapidly, but hard enough that naive approaches fail.

# What We Tried 

We implemented four approaches in pure NumPy (no frameworks), tracking operations, memory accesses, and reuse distance throughout. All approaches used the same dataset (1,000 training examples, 500 test) and the same network architecture (20 inputs, 32 hidden neurons, 1 output).

## Approach A: Pure Backpropagation 

Standard gradient descent baseline. The network attempts to learn which bits matter and how to compute XOR simultaneously through gradient updates. Result: accuracy plateaued around 60--68%, barely above the 50% coin-flip baseline. The gradients from the 3 secret bits are diluted by 17 noise bits, leaving the network unable to distinguish signal from noise within 200 epochs.

## Approach B: Pure Evolutionary 

A population of 100 random networks evaluated by forward pass only (no backward pass), with selection and mutation across 150 generations. Result: even worse at 52%, and used 37x more total operations than backprop. With 705 parameters per network, random mutation cannot efficiently search the weight space.

## Approach C: Naive Hybrid (Evolution then Backprop) 

Evolution to find a promising starting point, then backpropagation to refine. Result: 54%. The evolutionary phase never found a genuinely good candidate because it was searching 705-dimensional weight space, so it handed backprop a bad starting point that couldn't be rescued.

## Approach D: Mask-Based Hybrid 

Key insight: separate the search problem (which bits matter?) from the learning problem (what is their XOR?). Instead of evolving full networks (705 parameters), evolve a 20-number input mask that amplifies relevant bits and suppresses noise. Once the mask identifies the secret bits, backprop learns XOR on the filtered inputs easily. Result: 91--93% accuracy, under 2 seconds. The mask correctly identified all 3 secret bits in 2 out of 3 runs within the first 5--10 evolutionary generations.

## Approach E: Exhaustive Search + Backprop 

Since mask evaluation is fast when you know the function is XOR, we tested all 1,140 possible 3-bit combinations directly. The correct bits were found in 0.02 seconds with 100% search accuracy, and backprop then achieved 93% on the learning phase. Total time: 0.15 seconds. However, this approach \"cheats\" by hardcoding XOR knowledge into the evaluator and won't scale to larger problems.

# Results Comparison 

  ------------------------ ----------------------- ------------------------ --------------------------- -------------------------- -------------------------
  Metric      Backprop   Evolution   Naive Hybrid   Mask Hybrid   Exhaustive
  Accuracy         60--68%          \~52%             \~54%                91--93%             93%
  Time (sec)       0.23             1.27              0.29                 0.15\*              0.15
  Operations       271M             10B               1.95B                139M                142M
  Reuse Distance   9.7              3.3               3.3                  9.1                 9.5
  Passes 90%?      No               No                No                   Yes                 Yes
  ------------------------ ----------------------- ------------------------ --------------------------- -------------------------- -------------------------

\* Mask Hybrid time varies; Step 4 version took 17.5s before optimization, Step 5 optimized version runs in 0.15s.

# Key Insight: XOR's Flat Fitness Landscape 

The most important finding is why evolutionary search on bit-sets fails for XOR. When the secret bits are \[0, 3, 7\, guessing \0, 3, 12\ (2 out of 3 correct) scores exactly the same as guessing \5, 11, 16\ (0 correct): both get 50% accuracy. This is because XOR is perfectly balanced --- any missing bit flips the answer randomly, destroying all partial credit. The fitness landscape is completely flat with a single spike at the exact correct answer. This means evolution has no gradient to climb; it degenerates into random search.]

This property is specific to XOR/parity and wouldn't apply to most real learning tasks, where partial solutions typically produce partial improvements. However, it makes sparse parity an exceptionally hard benchmark for gradient-free methods, which is precisely what makes it useful as Yaroslav's \"drosophila\" --- any algorithm that solves it efficiently has demonstrated something genuinely powerful.

# Energy Observations 

Forward-only methods (evolution) achieved significantly better reuse distance (3.3) than backpropagation (9.7), confirming that eliminating the backward pass creates more cache-friendly memory access patterns. However, pure evolution required far more total operations, negating the per-operation efficiency gain. The mask-based hybrid found a middle ground: the evolutionary search phase is small (20 parameters, few generations), and the backprop refinement phase operates on filtered inputs where most weights are effectively zeroed out.

For the Sutro thesis, this suggests that decomposing learning into a lightweight search phase (forward-only, cache-friendly) followed by targeted refinement (backprop on a reduced problem) could offer better energy tradeoffs than either approach alone.

# Promising Next Steps 

Several directions are worth exploring:

1.  Scalable search on flat landscapes. The exhaustive approach works for 20-choose-3 (1,140 combinations) but breaks at 100-choose-5 (75 million). We need search strategies that work without fitness gradients: systematic sampling without replacement, information-theoretic bit scoring, or hierarchical elimination (test pairs/triples to narrow candidates before testing full subsets).
2.  Generalizable mask evaluation. Our fastest evaluator \"cheats\" by computing XOR directly. A more general version would use a fast proxy --- perhaps a very small network trained for just a few epochs --- to score masks without knowing the target function. The Step 4 version did this but was too slow; finding the right speed/accuracy tradeoff for the proxy evaluator is an open problem.
3.  Reuse distance optimization. Now that we have a working algorithm, we can focus on Yaroslav's energy metric. Restructuring the training loop to minimize average reuse distance --- for example, processing all data for one layer before moving to the next, or tiling computations to keep working sets small --- could reduce simulated energy cost without changing accuracy.
4.  Character-level prediction bridge. Yaroslav's ultimate goal is next-character prediction without tokenization. The mask-based decomposition (search for relevant features, then learn their relationship) could apply to character-level models where most input characters are noise relative to predicting the next one. This connection is worth developing.

::: c2
\[a\](#cmnt_ref1)I'm curious about the appraisal that was used. In other words, what did you start with? What were your initial prompts? And which things worked/failed
:::

::: c2
\[b\](#cmnt_ref2)I should probably show it to you this evening. It was a long chat.
:::

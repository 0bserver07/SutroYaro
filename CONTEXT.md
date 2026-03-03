# Context

## What is this?

A research environment for the Sutro Group's work on energy-efficient AI training. The group's thesis: go back to 1960s-era AI problems and reinvent learning algorithms using modern tools (AI agents, compute), with energy efficiency as the optimization target.

## Why Sparse Parity?

Sparse parity is the "drosophila" of learning tasks:
- Simplest non-trivial learning problem (XOR was the example Minsky used to trigger the AI winter)
- Easy to scale difficulty (add noise bits)
- Fast to iterate (<1 second training + eval)
- Exposes fundamental memory access patterns in backprop

## Key Insight from Sprint 1

Standard backprop has an inherent ARD bottleneck: parameter tensors (W1, b1) are read in forward pass and again at the end of backward pass, with the entire computation in between. Gradient fusion (fusing weight updates) only helps ~5% of total memory reads. Real improvement requires:
- Per-layer forward-backward without full network propagation
- Hinton's Forward-Forward algorithm
- Other non-backprop learning rules

## Timeline

- 19 Jan 2026: Meeting #1 - energy-efficient training intro
- 26 Jan 2026: Meeting #2 - Hinton's Forward-Forward
- 02 Feb 2026: Meeting #3 - Joules measuring
- 09 Feb 2026: Meeting #4 - From Beauty to Joules
- 16 Feb 2026: Meeting #5 - Intelligence Per Joule, Karpathy names task
- 23 Feb 2026: Meeting #6 - Germaine presentation, Emmett results
- 02 Mar 2026: Meeting #7 - Sparse parity task, Yaroslav sprint 1

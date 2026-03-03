# Technical Sprint 2 Plan

## Background
Sprint 1 showed that gradient fusion improves ARD by ~16%, but the real bottleneck is parameter tensors read twice across the full forward+backward pass. Fundamentally different algorithms are needed.

## Candidate Algorithms to Explore

1. **Hinton's Forward-Forward Algorithm** - Discussed in Meeting #2. Replaces backprop with local learning rules per layer.
2. **Per-layer forward-backward** - Compute each layer's backward and update before proceeding to next layer's forward. Changes the math but could drastically reduce ARD.
3. **Local learning rules** - Hebbian-style updates that don't require global backward pass.
4. **Gradient checkpointing variants** - Trade compute for memory in ways that reduce ARD.

## Approach

- [ ] Implement Forward-Forward on 3-bit parity as baseline
- [ ] Measure ARD and compare to standard backprop
- [ ] Try per-layer update scheme
- [ ] Scale to sparse parity (20 bits, 3 relevant)
- [ ] Document findings and prompting strategies

## Open Questions

- Does Forward-Forward converge on sparse parity?
- What's the theoretical minimum ARD for this task?
- Can we combine approaches?

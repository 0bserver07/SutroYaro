# Technical Sprint 2 Plan

## Background

Sprint 1 showed that gradient fusion improves ARD by ~16%, but the real bottleneck is parameter tensors read twice across the full forward+backward pass. Fundamentally different algorithms are needed.

## Candidate Algorithms

| Algorithm | Idea | Why it could help |
|-----------|------|-------------------|
| **Forward-Forward** | Two forward passes, no backward | Eliminates backward pass entirely |
| **Per-layer update** | Backward + update per layer before next forward | Keeps parameters in cache |
| **Local learning rules** | Hebbian-style, no global error signal | Maximum locality |
| **Gradient checkpointing** | Trade compute for memory | Reduces peak ARD |

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

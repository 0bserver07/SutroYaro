# Technical Sprint 2 Plan — COMPLETED

!!! note "Historical"
    This plan was written after Sprint 1 and has been fully executed. All candidate algorithms were tested. See the findings linked below.

## Background

Sprint 1 showed that gradient fusion improves ARD by ~16%, but the real bottleneck is parameter tensors read twice across the full forward+backward pass. Fundamentally different algorithms are needed.

## Candidate Algorithms — All Tested

| Algorithm | Result | Finding |
|-----------|--------|---------|
| **Forward-Forward** | 25x WORSE ARD, fails on 20-bit | [Exp E](../findings/exp_e_forward_forward.md) |
| **Per-layer update** | 3.8% ARD improvement, converges identically | [Exp C](../findings/exp_c_perlayer_20bit.md) |
| **Sign SGD** | Solves k=5, 2x faster than standard SGD | [findings](../findings/exp_sign_sgd.md) |
| **Curriculum learning** | 14.6x speedup on n=50/k=3 | [findings](../findings/exp_curriculum.md) |
| **Fourier solver** | 13x faster than SGD for small k | [findings](../findings/exp_fourier.md) |

## Completed Tasks

- [x] Implement Forward-Forward on 3-bit parity as baseline
- [x] Measure ARD and compare to standard backprop
- [x] Try per-layer update scheme
- [x] Scale to sparse parity (20 bits, 3 relevant)
- [x] Document findings and prompting strategies

## Answered Questions

- **Does Forward-Forward converge on sparse parity?** Only 3-bit. Fails on 20-bit due to greedy layer-wise objective.
- **What's the theoretical minimum ARD for this task?** W1 dominates at 75% of reads. Operation reordering capped at ~10% improvement.
- **Can we combine approaches?** Per-layer + batch works but isn't valuable. Single-sample SGD is 8x faster.

## What's Next

See [DISCOVERIES.md](https://github.com/0bserver07/SutroYaro/blob/main/DISCOVERIES.md) for open questions and [TODO.md](https://github.com/0bserver07/SutroYaro/blob/main/TODO.md) for remaining tasks.

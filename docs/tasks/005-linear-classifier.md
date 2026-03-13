# Task 5: Linear classifier for parity (arXiv:2309.06979)

**Priority**: MEDIUM
**Status**: REVIEWED - not directly applicable
**Source**: Telegram message #710 (reading group)

## Context

From the Telegram channel: "In my Thursday reading group we are covering https://arxiv.org/abs/2309.06979, which shows that a linear classifier can solve the parity task, so that's an alternative approach that's halfway between current world and blank slate."

This is interesting because:
- All 4 local learning rules (Hebbian, PC, EP, TP) failed at chance level
- A linear classifier that solves parity would mean there's a feature representation where parity IS linear
- Could have very good ARD (simple model, small working set)

## Review

The paper (Edelman et al., "The Impact of Reasoning Step Length on Large Language Models") is about auto-regressive next-token predictors with Chain-of-Thought (CoT). It shows linear models can compute parity when given intermediate reasoning tokens. This is a different setup from our benchmark:

- Our task: given 20 bits, predict the XOR of 3 secret bits in one shot
- Their task: auto-regressive prediction with intermediate CoT tokens

A linear classifier on raw {-1,+1} inputs cannot solve parity (E[y*x_i] = 0 for all i, including correct bits). This was confirmed in our exp_feature_select: pairwise and greedy selection provably fail.

The paper IS relevant to the "bigger picture" (nanoGPT axis), but not to the current sparse parity benchmark. Worth revisiting when we move to the next problem along Yaroslav's axis 3.

## Original tasks (deprioritized)

- [ ] Consider if CoT-style sequential prediction is testable on our benchmark
- [ ] If so, would need a modified data format with intermediate tokens
- [ ] More relevant for the nanoGPT final exam than for sparse parity

## References

- Paper: https://arxiv.org/abs/2309.06979
- Telegram message #710
- Our local learning failures: DISCOVERIES.md "Local Learning Rules" section
- Experiment template: src/sparse_parity/experiments/_template.py

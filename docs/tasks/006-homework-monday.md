# Task 6: Prep homework for next Monday

**Priority**: HIGH
**Status**: IN PROGRESS
**Source**: Meeting #8 homework, The Bigger Picture

## Context

From the Sutro Group main page (Meeting #8 homework):

> 1. Get agents to improve challenge #1: sparse parity (using ARD as the proxy for energy)
> 2. Present your results/your process/present your learnings (example, Yad's youtube video)

Yad already presented the first video. For next Monday we need:
- Show progress on the metric side (DMC, stack distance)
- Show any new experiment results
- Demonstrate the sync/collaboration workflow (Andy's PRs, runbook)

## Tasks

- [x] Complete task 1: DMC metric added. Baseline: ARD 4,104 / DMC 300,298
- [x] Complete task 2: Stack distance confirmed already implemented
- [x] Complete task 4: Germain depth-1 reproduced. ARD/float identical (0.36). Not a locality win, just smaller model.
- [x] Complete task 5: Linear classifier paper reviewed. CoT-based, not applicable to our one-shot benchmark.
- [ ] Review and merge Andy's PRs (#2, #3) - need gh auth to merge
- [ ] Prepare summary of what changed since Meeting #8
- [ ] Record a follow-up demo if appropriate

## What to present Monday

1. **DMC metric** - adopted from Yaroslav's Knowledge Sprint #2. Now tracking alongside ARD. Based on Ding et al. (arXiv:2312.14441). DMC = sum(sqrt(stack_distance)).
2. **Germain's depth-1 analysis** - reproduced the result, ARD drops 68% but so do total floats. ARD/float is identical. The improvement is from doing less work, not better locality.
3. **Collaboration workflow** - Andy opened 2 PRs and 1 issue. Reviewed as Claude Code. Sync runbook created for weekly/daily cadence.
4. **Task tracker** - 6 tasks extracted from Meeting #8 feedback, 4 completed, 1 reviewed, 1 in progress.

## References

- The Bigger Picture: docs/google-docs/bigger-picture.md
- Homework: "present your results/your process/present your learnings"
- First video: https://www.youtube.com/watch?v=h8dAU8yngxM

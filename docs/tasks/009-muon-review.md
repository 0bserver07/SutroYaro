# Task 9: Muon Optimizer Literature Review

**Priority**: MEDIUM
**Status**: IN PROGRESS
**Agent**: Antigravity
**Source**: Yaroslav's Information Bottleneck podcast interview (April 1), search_space.yaml (listed but untested)

## Context

Yaroslav mentioned Muon in the Information Bottleneck podcast as an example. The lab's learning-guide.md notes: "Muon (first optimizer to beat Adam in 10 years) was discovered on 2-second CIFAR runs."

Muon appears in `research/search_space.yaml` but was never tested. The question: does an optimizer that orthogonalizes gradients (Newton-Schulz iteration) reduce memory access patterns compared to Adam's moment tracking? Relevant to our ByteDMD metric.

## Tasks

- [ ] Read the Muon paper (https://kellerjordan.github.io/posts/muon/)
- [ ] Study ByteDMD metric (https://github.com/cybertronai/ByteDMD)
- [ ] Analyze whether Newton-Schulz iteration reduces byte-level data movement vs Adam
- [ ] Assess whether Muon helps on small networks (hidden=200) or only large LLMs
- [ ] Write findings to `findings/exp_muon_review.md` using the agent prompt scaffold

## References

- Agent prompt: [docs/agent-prompts/muon-review.md](../agent-prompts/muon-review.md)
- Muon paper: https://kellerjordan.github.io/posts/muon/
- ByteDMD: https://github.com/cybertronai/ByteDMD
- ByteDMD test cases: https://github.com/cybertronai/ByteDMD-examples
- Learning guide mention: [docs/learning-guide.md](../learning-guide.md)
- Search space: [research/search_space.yaml](../../research/search_space.yaml)

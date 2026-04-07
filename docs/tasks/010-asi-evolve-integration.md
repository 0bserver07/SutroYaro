# Task 10: ASI-Evolve Paper Review — Lessons for SutroYaro

**Priority**: HIGH
**Status**: IN PROGRESS
**Agents**: Kimi 2.5 (memory), Qwen 3.6 Plus (execution), Claude/GLM-5.1 (algorithms), Gemini (synthesis)
**Source**: G B's Telegram post (2026-04-06): "Seems relevant to https://arxiv.org/abs/2603.29640"

## Context

ASI-Evolve is a closed-loop autonomous AI-for-AI research framework combining learn-design-experiment-analyze cycles with a cognition base and analyzer. We already have the building blocks: evaluation harness (`harness.py`), ByteDMD metric, `DISCOVERIES.md` knowledge base, agent prompt system.

This is a **literature review**, not integration. Each agent independently extracts lessons from the paper relevant to our lab (memory systems, execution design, algorithm search). A synthesis agent combines findings into a report of potentially adoptable practices. No code changes, no pipeline construction.

## Tasks

- [ ] **Agent 1 (Systems Architect — Kimi)**: Read `docs/agent-prompts/asi-evolve/memory-kimi.md`, follow instructions, write to `docs/research/asi-evolve/kimi_asi_memory.md`
- [ ] **Agent 2 (Implementation Engineer — Qwen)**: Read `docs/agent-prompts/asi-evolve/execution-qwen.md`, follow instructions, write to `docs/research/asi-evolve/qwen_asi_execution.md`
- [ ] **Agent 3 (Algorithms Expert — Claude/GLM)**: Read `docs/agent-prompts/asi-evolve/algorithms-claude.md`, follow instructions, write to `docs/research/asi-evolve/claude_asi_algorithms.md`
- [ ] **Agent 4 (Synthesis — Gemini)**: Read all 3 findings from above, follow `docs/agent-prompts/asi-evolve/synthesis-gemini.md`, write to `docs/research/asi-evolve/integration-plan.md`

## References

- Paper: https://arxiv.org/abs/2603.29640
- ByteDMD metric: https://github.com/cybertronai/ByteDMD
- ByteDMD examples: https://github.com/cybertronai/ByteDMD-examples
- Wesley Smith meeting notes: [docs/google-docs/31mar26-meeting-wesley-smith.md](docs/google-docs/) (pebble game, Strassen's)
- Agent prompts: [docs/agent-prompts/asi-evolve-*](../agent-prompts/)
- DISCOVERIES.md: what's already known (Hebbian failure, GF(2), KM-min)
- AGENT.md: current autonomous loop protocol

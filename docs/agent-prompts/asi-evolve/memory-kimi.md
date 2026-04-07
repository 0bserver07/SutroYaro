# ASI-Evolve: Memory & Analysis Architecture
**Agent**: Kimi 2.5 | **Role**: Systems Architect (Cognition Base + Analyzer)

Check out this project in your current working directory. Read `DISCOVERIES.md`, `CLAUDE.md`, `LAB.md`, and `AGENT.md` first.

Then, read the paper 'ASI-Evolve: AI Accelerates AI' (arXiv:2603.29640). Focus strictly on the 'Learn' and 'Analyze' phases of their framework—specifically how they implement the 'cognition base' to inject human priors, and the 'dedicated analyzer' to distill experimental outcomes into reusable insights.

Write a report titled `docs/research/asi-evolve/kimi_asi_memory.md` detailing exactly how we can map the ASI-Evolve 'Analyzer' and 'Cognition Base' abstractions onto our existing SutroYaro markdown files.

**CRITICAL METRIC CONTEXT**: We evaluate success using the ByteDMD metric (Data Movement Distance). Read the implementation at https://github.com/cybertronai/ByteDMD and the test cases at https://github.com/cybertronai/ByteDMD-examples. The metric models memory as an LRU stack and penalizes reads based on the square root of their stack depth (C = ∑ ceil(√D(b)) per byte). The repo shows how the metric was hardened against agents bypassing the wrapper. How do we structure the 'Cognition Base' to prevent the agent from repeatedly proposing algorithms that fail this specific spatial locality constraint?

**CRITICAL CONSTRAINT**: This is an analysis only. DO NOT modify any lab files. DO NOT modify `DISCOVERIES.md`, `LAB.md`, `CLAUDE.md`, `AGENT.md`, or any `src/` files. Write only to `docs/research/asi-evolve/kimi_asi_memory.md`.

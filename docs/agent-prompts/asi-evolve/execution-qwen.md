# ASI-Evolve: Execution Pipeline Design
**Agent**: Qwen 3.6 Plus via Qwen-Code | **Role**: Implementation Engineer (Design + Experiment loop)

Check out this project in your current working directory. Read `DISCOVERIES.md`, `CLAUDE.md`, `LAB.md`, and `AGENT.md` first.

Then, read the paper 'ASI-Evolve: AI Accelerates AI' (arXiv:2603.29640). Focus strictly on the 'Design' and 'Experiment' execution loop.

We have a locked evaluation harness (`src/harness.py`) and a Gymnasium environment (`SutroYaro/SparseParity-v0`) that grades agents on solving sparse parity using the ByteDMD metric (which wraps Python objects to track LRU stack depth). Read the ByteDMD implementation at https://github.com/cybertronai/ByteDMD for details on how the metric was hardened against agents bypassing the wrapper.

Write a report in `docs/research/asi-evolve/qwen_asi_execution.md` that outlines the orchestration needed to wrap our existing Gym environment in an ASI-Evolve loop. How does the agent parse the specific ByteDMD stack trace output to figure out *why* an algorithm was penalized?

**CRITICAL CONSTRAINT**: This is a design analysis only. DO NOT write code. DO NOT modify `harness.py` or any `src/` files. Write only to `docs/research/asi-evolve/qwen_asi_execution.md`.

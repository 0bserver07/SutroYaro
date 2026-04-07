# ASI-Evolve: Algorithm Search Strategy
**Agent**: Claude Code / GLM-5.1 | **Role**: Algorithms Expert (Architecture + RL Algorithm Design)

Check out this project in your current working directory. Read `DISCOVERIES.md`, `CLAUDE.md`, `LAB.md`, and `AGENT.md` first.

Then, read the paper 'ASI-Evolve: AI Accelerates AI' (arXiv:2603.29640). Focus strictly on how the framework achieved SOTA results in 'reinforcement learning algorithm design' and 'neural architecture design.'

Our target metric is ByteDMD. Read the implementation at https://github.com/cybertronai/ByteDMD and the test cases at https://github.com/cybertronai/ByteDMD-examples. Traditional algorithms like SGD fail miserably because ByteDMD penalizes memory reads based on their depth in an LRU stack (C = ∑ ceil(√D(b)) per byte). Moving gradients back and forth destroys the score.

Write a report in `docs/research/asi-evolve/claude_asi_algorithms.md` analyzing the evolutionary search strategies ASI-Evolve used. What specific mutation operators or search-space constraints should we program into our agent so it invents a novel learning rule that achieves high spatial locality (keeping active variables at the top of the LRU stack) to minimize ByteDMD?

**CRITICAL CONSTRAINT**: This is an analysis only. DO NOT run experiments. Write only to `docs/research/asi-evolve/claude_asi_algorithms.md`.

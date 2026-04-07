# Gemini Review: ASI-Evolve Task Setup

I have set up Task 10: ASI-Evolve integration into SutroYaro workflow. Review the following files I created and check:

1. **Completeness**: Are all 4 agent prompts present? Do they map to the 4 roles (Kimi=memory, Qwen=execution, Claude=algorithms, Gemini=synthesis)?
2. **Guardrails**: Each prompt has constraints (no code modification, no experiment runs). Are they sufficient?
3. **ByteDMD context**: Each prompt references https://github.com/cybertronai/ByteDMD and https://github.com/cybertronai/ByteDMD-examples. Is the formula correct?
4. **Synthesis prompt**: Does the Gemini synthesis prompt include the Wesley Smith pebble game reference, the low-hanging fruit vs architectural changes distinction, and the ByteDMD constraints?
5. **Task spec**: Is `docs/tasks/010-asi-evolve-integration.md` complete enough?

Files to review:
- `docs/tasks/010-asi-evolve-integration.md`
- `docs/agent-prompts/asi-evolve-memory-kimi.md`
- `docs/agent-prompts/asi-evolve-execution-qwen.md`
- `docs/agent-prompts/asi-evolve-algorithms-claude.md`
- `docs/agent-prompts/asi-evolve-synthesis-gemini.md`
- `docs/tasks/INDEX.md` (should have rows 9 and 10)

Report any issues, missing constraints, or improvements.

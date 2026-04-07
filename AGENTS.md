# AGENTS.md

This project uses AI agents (Claude Code, Codex CLI, Gemini CLI, Replit, others) for research and accepts contributions from both humans and agents.

## For single agents (getting started)

If you are the only agent working on this project, read these in order:

1. **CLAUDE.md** - Deep technical context: project goals, metrics (ARD/DMC), best methods, 36 experiments, current state
2. **DISCOVERIES.md** - Knowledge base: proven facts, failed approaches, open questions (Q7, Q11-Q13)
3. **LAB.md** - Experiment protocol: templates, lifecycle, baselines, rules
4. **AGENT.md** (optional) - Only if running autonomous overnight loops

**Note:** CLAUDE.md is the canonical technical source regardless of which model/tool you use (Claude, Gemini, Kimi, etc.).

## How AI agents were used

**Phase 1** (16 experiments): Single Claude Code sessions running experiments sequentially, each following the template in `src/sparse_parity/experiments/_template.py`.

**Phase 2** (17 experiments): 17 independent Claude Code agents dispatched in parallel, each implementing a different algorithmic approach to sparse parity. Each agent received the approach description, shared module APIs, three test configs, and a findings template. All 17 completed successfully, producing code, results, and findings.

**Survey**: A single agent wrote the Practitioner's Field Guide (`docs/research/survey.md`) synthesizing all 33 experiments, with spec compliance and code quality review passes.

**Meeting #8 onward**: Multiple group members run their own agent harnesses (Germain's Replit Research OS, Michael's Claude approach, Yaroslav's Gemini). Results flow in via PRs and the `contributions/` directory.

## For agents reviewing PRs

When reviewing a contributed experiment:

1. Check that the experiment ran (results.json should match the findings doc)
2. Check DISCOVERIES.md for prior work on the same question
3. Verify the contributor didn't modify measurement code (tracker.py, cache_tracker.py, data.py, config.py)
4. If the experiment answers an open question, check that DISCOVERIES.md is updated

## For agents running experiments

1. Read DISCOVERIES.md first
2. Follow the template in `src/sparse_parity/experiments/_template.py`
3. Report both ARD and DMC (Data Movement Complexity)
4. Write findings to `findings/exp_{name}.md` using `findings/_template.md`
5. Never modify measurement code (LAB.md rule #9)

## Key files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project context for Claude Code (auto-loaded at session start) |
| `CODEX.md` | Project context for Codex CLI (auto-loaded via `.codex/config.toml`) |
| `.codex/AGENTS.md` | Codex instructions: context loading, sync routine, writing rules |
| `LAB.md` | Experiment protocol (one hypothesis, baseline, commit discipline) |
| `DISCOVERIES.md` | Shared knowledge base, anyone can PR new findings |
| `CONTRIBUTING.md` | How humans and agents contribute (three effort levels) |
| `contributions/` | Drop raw results here in any format |
| `findings/_template.md` | Standalone findings template |
| `docs/tasks/INDEX.md` | Current task tracker with priorities |
| `docs/research/survey.md` | Full methodology in Section 7 (agentic loop, parallel dispatch, prompting strategies) |
| `docs/tooling/anti-slop-guide.md` | Writing rules applied to all agent-generated prose |
| `docs/tooling/sync-runbook.md` | Weekly/daily/per-session sync checklists |

## What worked

- **DISCOVERIES.md as shared memory**: No agent repeated known-bad configurations (LR=0.5, GrokFast).
- **"It's OK to fail" prompts**: Agents produced better negative results when told to document failure reasons rather than forced to succeed.
- **Strict output format**: All 33 experiments are directly comparable because every agent used the same structure.
- **Literature-first prompting**: Giving agents theoretical context ("parity is linear over GF(2)") led to correct implementations.

## What didn't work

- **Pyright import warnings**: Every agent triggered false "Import could not be resolved" errors (PYTHONPATH is set at runtime).
- **Data generation bugs**: Two agents (Hebbian, Binary Weights) used different seeds for train/test. Fixed during execution.
- **Pebble Game exhaustive search**: One agent took 38 minutes exploring 5,758 topological orderings. A smarter search heuristic would have been faster.

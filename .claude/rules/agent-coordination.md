# Agent Coordination

Rules for dispatching parallel agents and managing file ownership.

## When to parallelize

Only dispatch agents in parallel when they write to non-overlapping files. If two agents might edit the same file, run them sequentially.

Before dispatching, list the files each agent will create or edit. If any file appears in more than one agent's list, do not parallelize.

## File ownership

| Agent type | Owns (can create/edit) | Must not touch |
|-----------|----------------------|---------------|
| Experiment agents | `src/sparse_parity/experiments/`, `results/{exp_id}/` | harness.py, tracker.py, config.py, data.py, cache_tracker.py |
| Findings agents | `docs/findings/` | `findings/` at repo root is wrong -- use `docs/findings/` |
| Sync agents | `docs/google-docs/`, `src/sparse_parity/telegram_sync/` | |
| Eval agents | `src/sparse_parity/eval/`, `results/eval/` | |
| Docs agents | `docs/` (non-findings, non-google-docs) | |
| Infrastructure agents | `src/sparse_parity/` (non-locked files), `.claude/` | |

## After parallel agents complete

The main agent must verify all outputs before committing:

1. Check that files are in the correct directories (not repo root when they should be in docs/)
2. Check that no locked files were modified
3. Run `python3 -m mkdocs build` if any docs were changed
4. Run experiments if code was changed

Do not auto-commit from sub-agents. The main agent reviews and commits.

## Dependency ordering

If task B needs results from task A, never run them in parallel. Check the dependency before dispatching.

Example: "run baseline sweep" must complete before "optimize based on baseline results." These are sequential even though they could technically be dispatched to different agents.

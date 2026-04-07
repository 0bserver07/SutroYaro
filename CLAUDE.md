# CLAUDE.md - Sutro Group Research Workspace

## Project Context

This is a research workspace for the **Sutro Group**, a study group exploring energy-efficient AI training. The group meets weekly at South Park Commons in San Francisco.

## Read These First

- **LAB.md** — Protocol for running experiments (templates, lifecycle, rules)
- **AGENT.md** — Machine-executable experiment loop for autonomous sessions
- **DISCOVERIES.md** — What's proven so far (read before every experiment)
- **CONTRIBUTING.md** — How external contributors submit experiments and findings
- **TODO.md** — Open research tasks
- **docs/tasks/INDEX.md** — Current task tracker with priorities
- **docs/research/survey.md** — Practitioner's Field Guide ranking all 36 experiments
- **docs/research/peer-research-protocol.md** — Full design doc for multi-researcher autonomous research

## Core Concepts

- **Sparse Parity**: The benchmark task — learn XOR/parity from random {-1,+1} inputs. n=20 bits, k=3 secret, 17 noise. The "drosophila" of energy-efficient training.
- **Average Reuse Distance (ARD)**: Proxy metric for energy efficiency. Small ARD = data stays in cache = cheap. Large ARD = expensive external memory access.
- **Data Movement Complexity (DMC)**: Better proxy metric (Ding et al., arXiv:2312.14441). DMC = sum of sqrt(stack_distance) for all float accesses. Tracks alongside ARD in MemTracker. Baseline: ARD 4,104 / DMC 300,298.
- **Cache Energy Model**: register 5pJ, L1 (64KB) 20pJ, L2 (256KB) 100pJ, HBM 640pJ per float access (Bill Dally numbers).
- **CacheTracker**: Extended MemTracker with LRU cache simulation for realistic energy estimates.
- **TrackedArray / Auto DMD**: `TrackedArray` wraps numpy arrays so every operation (ufuncs, indexing, slicing) auto-records reads and writes on an `LRUStackTracker`. Removes manual instrumentation errors. See `docs/research/tracked-numpy.md`.

## Current Best Methods

| Method | Time (n=20/k=3) | ARD | DMC | Notes |
|--------|-----------------|-----|-----|-------|
| KM-min (1 sample) | ~0.001s | 20 | 3,578 | New DMC leader. 1 influence sample suffices for parity. |
| GF(2) Gaussian Elimination | 509 us | ~420 | ~203K | 240x faster than SGD, k-independent. Auto-tracked via TrackedArray; old harness reported 8,607. |
| KM Influence Estimation | 0.001-0.006s | 92 | 20,633 | ARD leader. 5 influence samples per bit. |
| SMT Backtracking | 0.002s | 3,360 | 348,336 | Constraint satisfaction approach |
| SGD (baseline) | 0.12s | 8,504 | 1,278,460 | LR=0.1, batch=32, hidden=200 |

GF(2) solves n=100/k=10 in 703 microseconds. Parity is linear over the binary field -- the neural network was solving an easy problem the hard way.

## SGD Config (when using neural nets)

```python
n_bits=20, k_sparse=3, hidden=200, lr=0.1, wd=0.01,
batch_size=32, n_train=1000, max_epochs=200
```

Solves in ~40 epochs / 0.12s with numpy (`fast.py`).

## Nix Development Shell (Optional)

For NixOS users (or those with flakes), a `flake.nix` provides a reproducible environment with python3 + numpy. Non-NixOS users can ignore the nix files.

```bash
nix develop
python3 bin/reproduce-all
```

Or one-liner:
```bash
nix develop --command python3 bin/reproduce-all
```

For reproducibility, macOS/Linux users can install Nix via the Determinate Systems installer:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://install.determinate.systems/nix | sh -s -- install
```

This is informational (tells agents nix is available) rather than controlling (instructing behavior). Non-NixOS users can ignore and run `python3` directly.

## Key Findings

**Phase 1 (16 experiments, SGD optimization):**
- LR=0.1 is critical (0.5 overshoots, never triggers phase transition)
- W1 dominates 75% of all float reads -- limits ARD optimization to ~10%
- L2 cache (256KB) eliminates ALL cache misses for both single-sample and batch
- Curriculum learning (n=10 then expand to n=50) gives 14.6x speedup at scale
- SGD breaks when n^k exceeds ~100,000 gradient steps

**Phase 2 (17 experiments, broad search):**
- Algebraic/exact methods (GF(2), KM, SMT) solve instantly -- they exploit that parity is linear over GF(2)
- All 4 local learning rules (Hebbian, Predictive Coding, Equilibrium Propagation, Target Propagation) fail at chance level -- parity requires k-th order interaction detection
- Information-theoretic methods (MI, LASSO, MDL, Random Projections) all solve it but none beats Fourier meaningfully
- RL sequential Q-learning achieves ARD of 1 at inference (reads exactly k=3 bits per prediction)

## Autonomous Research Infrastructure

| File | Purpose |
|------|---------|
| `AGENT.md` | Agent-executable experiment loop (machine protocol) |
| `src/harness.py` | Locked evaluation harness (DO NOT MODIFY in experiment PRs) |
| `research/search_space.yaml` | Bounded mutation space per challenge |
| `research/questions.yaml` | Dependency graph of open research questions |
| `research/log.jsonl` | Append-only experiment log (machine-readable) |
| `results/scoreboard.tsv` | Human-readable leaderboard (auto-generated) |
| `checks/env_check.py` | Pre-flight environment check |
| `checks/baseline_check.py` | Re-establish baselines on this machine |
| `bin/run-agent` | Launch autonomous agent cycle |
| `bin/merge-findings` | Import contributor log entries via PR |

See [docs/research/peer-research-protocol.md](docs/research/peer-research-protocol.md) for the full design.

## Eval Environment

The eval environment tests whether an AI agent can do energy-efficient ML research.

- **Guide**: See `AGENT_EVAL.md` for adding challenges, methods, running evals
- **Quick test**: `PYTHONPATH=src python3 -c "import gymnasium as gym; import sparse_parity.eval; env = gym.make('SutroYaro/SparseParity-v0', metric='dmc', budget=10); obs, info = env.reset(); obs, r, _, _, info = env.step(5); print(info)"`
- **Environments**: `SutroYaro/SparseParity-v0` (single challenge), `SutroYaro/MultiChallenge-v0` (all three)
- **Ground truth**: 36 experiments, 72-point grading rubric (12 categories)
- **Docs**: `docs/research/eval-environment.md`

## Agent Infrastructure

Hooks, rules, and skills that make Claude Code sessions more effective. See [docs/research/agent-infrastructure.md](docs/research/agent-infrastructure.md) for the full docs.

- **Hooks**: session-start (shows status), security-guard (blocks locked measurement code and destructive commands), session-end (session summary)
- **Rules**: experiment reproducibility (seeds, config, environment), agent coordination (parallel dispatch, file ownership)
- **Skills**: run-experiment (two-phase protocol), weekly-catchup, prepare-meeting
- **Config**: `.claude/settings.json`

Other coding agents (Gemini, Codex) don't run the hooks but can read the rules and skills.

## Automation

| Script | What it does | Docs |
|--------|-------------|------|
| `bin/tg-sync` | Syncs Telegram to local SQLite (incremental) | [docs/tooling/telegram-setup.md](docs/tooling/telegram-setup.md) |
| `bin/tg-post` | Posts to Telegram forum topics via Bot API | [docs/tooling/telegram-setup.md](docs/tooling/telegram-setup.md) |
| `src/sync_google_docs.py` | Pulls Google Docs to local markdown | [docs/tooling/automation.md](docs/tooling/automation.md) |
| `.traces/export_sessions.py` | Exports Claude Code session traces | [docs/tooling/automation.md](docs/tooling/automation.md) |

### Telegram Quick Reference

```bash
# First time: install deps and authenticate
bun install
# On NixOS: credentials come from sops-nix via flake shellHook, no .env needed
# On other systems: cp .env.example .env and fill in credentials
bin/tg-auth

# Sync messages to SQLite (incremental)
bin/tg-sync
# Database: telegram.db (project root, .gitignored)

# Query messages
sqlite3 telegram.db "SELECT date, sender, text FROM messages ORDER BY date DESC LIMIT 10"

# Post via your own bot (requires TELEGRAM_BOT_TOKEN in .env)
bin/tg-post --topic agent-updates "Experiment completed"
```

## Working Style

- Iteration time must stay under 2 seconds (use `fast.py` for numpy speed)
- Change one thing at a time (correctness, then speed, then energy)
- Priority: correctness > wall-clock time > energy usage
- One hypothesis per experiment, always compare against baseline
- Record everything -- failed hypotheses are findings too
- Apply anti-slop writing rules to all prose (no em dashes, no AI vocabulary)

## Before Pushing

- **Update `docs/changelog.md`** with what changed (bump version, add section)
- **Sync Google Docs** if meeting notes may have changed: `python3 src/sync_google_docs.py`
- **Sync Telegram** if group discussion may have new messages: `bun run sync_telegram.ts`
- **Check `docs/index.md`** if findings or status changed -- homepage should reflect current state
- **Check GitHub** for PRs/issues: `gh pr list --repo cybertronai/SutroYaro`

Full sync workflow: [docs/tooling/sync-runbook.md](docs/tooling/sync-runbook.md)

## People

- **Yad Konrad** (@yad) — repo creator, SutroYaro. Built the Claude Code autonomous research lab, parallel agent experiments
- **Yaroslav Bulatov** (@yaroslavvb) — Sutro Group founder. Technical sprints, algorithm work, cybertronai/sutro
- **Emmett** — Aster agentic loop framework, 2x energy improvement on microgpt
- **G B** — Architecture experiments (depth-1/hidden-64, ARD ~33-35)
- **Germaine**, **Andy Zhang** (@zh4ng), **Seth Stafford** (@sethts), **Barak**, **Jamie Simon** — Group members

## Contributing

Multiple people contribute via PRs (fork and branch). See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide and [docs/branch-workflow.md](docs/branch-workflow.md) for branch naming, locked files, and agent permissions.

- **`contributions/`** — Drop raw results here in any format. No template needed.
- **`findings/_template.md`** — Standalone findings template for structured reports.
- **`DISCOVERIES.md`** — Shared knowledge base. Anyone can PR new bullets.
- **Metric isolation (LAB.md rule #9)** — Never modify tracker.py, cache_tracker.py, data.py, config.py, harness.py in experiment PRs.

When reviewing PRs: check that results are reproducible, findings follow the template, and DISCOVERIES.md is updated if the experiment answers an open question.

## Related Repos

- https://github.com/cybertronai/sutro — Main code repo with sparse_parity_benchmark.py
- https://github.com/cybertronai/SutroYaro — This research workspace

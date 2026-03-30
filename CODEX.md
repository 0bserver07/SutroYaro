# CODEX.md - Sutro Group Research Workspace

## Project Context

This is a research workspace for the **Sutro Group**, a study group exploring energy-efficient AI training. The group meets weekly at South Park Commons in San Francisco.

## Read These First

- **LAB.md** — Protocol for running experiments (templates, lifecycle, rules)
- **AGENT.md** — Machine-executable experiment loop for autonomous sessions
- **DISCOVERIES.md** — What's proven so far (read before every experiment)
- **CONTRIBUTING.md** — How external contributors submit experiments and findings
- **TODO.md** — Open research tasks
- **docs/tasks/INDEX.md** — Current task tracker with priorities
- **docs/research/survey.md** — Practitioner's Field Guide ranking all 33 experiments
- **docs/research/peer-research-protocol.md** — Full design doc for multi-researcher autonomous research

## Core Concepts

- **Sparse Parity**: The benchmark task — learn XOR/parity from random {-1,+1} inputs. n=20 bits, k=3 secret, 17 noise. The "drosophila" of energy-efficient training.
- **Average Reuse Distance (ARD)**: Proxy metric for energy efficiency. Small ARD = data stays in cache = cheap. Large ARD = expensive external memory access.
- **Data Movement Complexity (DMC)**: Better proxy metric (Ding et al., arXiv:2312.14441). DMC = sum of sqrt(stack_distance) for all float accesses. Tracks alongside ARD in MemTracker. Baseline: ARD 4,104 / DMC 300,298.
- **Cache Energy Model**: register 5pJ, L1 (64KB) 20pJ, L2 (256KB) 100pJ, HBM 640pJ per float access (Bill Dally numbers).
- **CacheTracker**: Extended MemTracker with LRU cache simulation for realistic energy estimates.

## Current Best Methods

| Method | Time (n=20/k=3) | ARD | DMC | Notes |
|--------|-----------------|-----|-----|-------|
| KM-min (1 sample) | ~0.001s | 20 | 3,578 | New DMC leader. 1 influence sample suffices for parity. |
| GF(2) Gaussian Elimination | 509 us | ~420 | 8,607 | 240x faster than SGD, k-independent. Harness under-counts; true DMC ~189K. |
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

## Automation

| Script | What it does | Docs |
|--------|-------------|------|
| `sync_telegram.ts` | Bulk-syncs Telegram topics to JSON files | [docs/tooling/automation.md](docs/tooling/automation.md) |
| `telegram/tg-topics.ts` | Lists forum topics (JSON) | See Telegram section below |
| `telegram/tg-read.ts` | Reads messages from a topic (JSON) | See Telegram section below |
| `telegram/tg-send.ts` | Sends a message to a topic | See Telegram section below |
| `src/sync_google_docs.py` | Pulls Google Docs to local markdown | [docs/tooling/automation.md](docs/tooling/automation.md) |
| `bin/review-cycle` | Cross-model experiment review (supervisor/researcher dialogue) | See below |

### Review Cycle Quick Reference

```bash
# Codex supervises Claude's work (last 5 experiments, 3-turn dialogue)
bin/review-cycle --tool codex --researcher-tool claude --last 5

# Claude supervises Codex's work
bin/review-cycle --tool claude --researcher-tool codex --last 5

# Preview prompts without launching agents
bin/review-cycle --dry-run --last 3

# Output: research/reviews/review-{timestamp}.md
```

### Telegram Quick Reference

```bash
# First time: install deps and authenticate
bun install
cp .env.example .env  # fill in TELEGRAM_API_ID and TELEGRAM_API_HASH
tg auth login

# Bulk sync (existing)
bun run sync_telegram.ts

# List topics
bun telegram/tg-topics.ts

# Read last 20 messages from a topic
bun telegram/tg-read.ts --topic "General" --limit 20

# Read messages since a date
bun telegram/tg-read.ts --topic "chat-yad" --since 2025-06-01

# Send a message to a topic
bun telegram/tg-send.ts --topic "agents" --message "Hello from agent"

# Send multi-line via stdin
echo "Summary of findings..." | bun telegram/tg-send.ts --topic "agents" --stdin

# Send to default write topic (set TELEGRAM_WRITE_TOPIC in .env)
bun telegram/tg-send.ts --message "Status update"
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

- **Yad** (repo creator, SutroYaro) — Built the Claude Code autonomous research lab, parallel agent experiments
- **Yaroslav** (Sutro Group founder) — Technical sprints, algorithm work, cybertronai/sutro
- **Emmett** — Aster agentic loop framework, 2x energy improvement on microgpt
- **G B** — Architecture experiments (depth-1/hidden-64, ARD ~33-35)
- **Germaine**, **Andy**, **Seth**, **Barak**, **Jamie Simon** — Group members

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

# CLAUDE.md - Sutro Group Research Workspace

## Project Context

This is a research workspace for the **Sutro Group**, a study group exploring energy-efficient AI training. The group meets weekly at South Park Commons in San Francisco.

## Read These First

- **LAB.md** — Protocol for running experiments (templates, lifecycle, rules)
- **DISCOVERIES.md** — What's proven so far (read before every experiment)
- **TODO.md** — Open research tasks
- **docs/research/survey.md** — Practitioner's Field Guide ranking all 33 experiments

## Core Concepts

- **Sparse Parity**: The benchmark task — learn XOR/parity from random {-1,+1} inputs. n=20 bits, k=3 secret, 17 noise. The "drosophila" of energy-efficient training.
- **Average Reuse Distance (ARD)**: Proxy metric for energy efficiency. Small ARD = data stays in cache = cheap. Large ARD = expensive external memory access.
- **Cache Energy Model**: register 5pJ, L1 (64KB) 20pJ, L2 (256KB) 100pJ, HBM 640pJ per float access (Bill Dally numbers).
- **CacheTracker**: Extended MemTracker with LRU cache simulation for realistic energy estimates.

## Current Best Methods

| Method | Time (n=20/k=3) | ARD | Notes |
|--------|-----------------|-----|-------|
| GF(2) Gaussian Elimination | 509 us | ~500 | 240x faster than SGD, k-independent |
| KM Influence Estimation | 0.001-0.006s | 1,585 | 724x better ARD than Fourier |
| SMT Backtracking | 0.002s | ~2,000 | Constraint satisfaction approach |
| SGD (baseline) | 0.12s | 17,976 | LR=0.1, batch=32, hidden=200 |

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

## Automation

| Script | What it does | Docs |
|--------|-------------|------|
| `sync_telegram.ts` | Pulls Telegram group thread messages to JSON | [docs/tooling/automation.md](docs/tooling/automation.md) |
| `src/sync_google_docs.py` | Pulls Google Docs to local markdown | [docs/tooling/automation.md](docs/tooling/automation.md) |
| `.traces/export_sessions.py` | Exports Claude Code session traces | [docs/tooling/automation.md](docs/tooling/automation.md) |

### Telegram Sync Quick Reference

```bash
# First time: install deps and authenticate
bun install
cp .env.example .env  # fill in TELEGRAM_API_ID and TELEGRAM_API_HASH
tg auth login

# Sync messages
bun run sync_telegram.ts
# Output: src/sparse_parity/telegram_sync/messages.json
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

## People

- **Yad** (repo creator, SutroYaro) — Built the Claude Code autonomous research lab, parallel agent experiments
- **Yaroslav** (Sutro Group founder) — Technical sprints, algorithm work, cybertronai/sutro
- **Emmett** — Aster agentic loop framework, 2x energy improvement on microgpt
- **G B** — Architecture experiments (depth-1/hidden-64, ARD ~33-35)
- **Germaine**, **Andy**, **Seth**, **Barak**, **Jamie Simon** — Group members

## Related Repos

- https://github.com/cybertronai/sutro — Main code repo with sparse_parity_benchmark.py
- https://github.com/0bserver07/SutroYaro — This research workspace

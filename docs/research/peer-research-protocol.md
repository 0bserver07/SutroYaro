# Peer Research Protocol

How SutroYaro runs autonomous, multi-researcher experiments on shared challenges.

## Why this exists

At Meeting #8 (09 Mar 2026), four people independently ran agent-driven research on the same sparse parity challenge using different tools (Claude Code, Replit, Gemini, plain Python). Each produced valid results. One researcher's agents rewrote the measurement code to inflate scores instead of improving the algorithm.

This protocol solves three problems:

1. **Comparability**: a locked evaluation harness means everyone measures the same way
2. **Accumulation**: a machine-readable experiment log means results can be queried, merged, and built upon across researchers
3. **Autonomy**: an agent-executable loop means any researcher can run overnight experiments without babysitting

## Two layers

### Layer 1: Single-agent loop

Each researcher (or their agent) runs experiments against a locked harness. The loop:

```
1. Read DISCOVERIES.md (what's known)
2. Read research/questions.yaml (what's open)
3. Pick top unchecked item from TODO.md
4. Design a single-variable experiment
5. Run against src/harness.py (locked, cannot modify)
6. Log result to research/log.jsonl
7. Classify: WIN / LOSS / INVALID / INCONCLUSIVE
8. Update TODO.md checkbox
9. Repeat until queue empty or interrupted
```

The agent-executable version lives in `AGENT.md`. The human protocol stays in `LAB.md`. Both produce the same output format.

### Layer 2: Peer merge

Multiple researchers submit findings via PR. The merge process:

```
Yad's machine              Germain's machine           Yaroslav's machine
+------------------+       +------------------+        +------------------+
| AGENT.md loop    |       | Replit agents    |        | Gemini / manual  |
| log.jsonl (local)|       | log.jsonl (local)|        | log.jsonl (local)|
| harness.py (locked)      | harness.py (locked)       | harness.py (locked)
+--------+---------+       +--------+---------+        +--------+---------+
         |                          |                            |
         | PR                       | PR                         | PR
         +----------+---------------+----------------------------+
                    v
         +---------------------+
         | Shared repo          |
         | DISCOVERIES.md       | <-- merged findings
         | research/log.jsonl   | <-- merged experiment records
         | results/scoreboard.tsv <-- auto-generated
         | research/questions.yaml <-- updated
         +---------------------+
```

Each researcher:

1. Forks or branches the repo
2. Runs their agent loop (any tool)
3. Produces a local `research/log.jsonl` using the shared schema
4. Submits a PR with log entries + findings docs
5. `bin/merge-findings` deduplicates and integrates

The locked `src/harness.py` makes results comparable. Everyone measures ARD/DMC/timing the same way.

## File map

```
AGENT.md                     # Agent-executable experiment loop (new)
LAB.md                       # Human protocol (unchanged)
DISCOVERIES.md               # Shared knowledge base (unchanged)
CONTRIBUTING.md              # How to submit results (unchanged)
TODO.md                      # Hypothesis queue with checkboxes (restructured)

src/
  harness.py                 # Locked evaluation: ARD, DMC, timing, baselines
                             # Rule #9: agents CANNOT modify this file

research/
  search_space.yaml          # What the agent can vary, per challenge
  questions.yaml             # Dependency graph of open questions
  log.jsonl                  # Append-only experiment records

results/
  scoreboard.tsv             # Human-readable leaderboard (from log.jsonl)
  progress.png               # ARD progress chart (from bin/analyze-log --plot)

checks/
  env_check.py               # Pre-flight: imports, fast.py, data generation
  baseline_check.py          # Re-establish baselines on this machine

bin/
  run-agent                  # Launch autonomous cycle
  merge-findings             # Import contributor log entries
  analyze-log                # Progress report + chart generation
```

## Running autonomously

Three modes, from simplest to most resilient. All are tool-agnostic (Claude, Gemini, or any CLI).

**Single cycle** -- one AI session, up to N experiments:
```bash
bin/run-agent --max 20
bin/run-agent --max 20 --tool gemini
```

**Looped mode** (recommended for overnight) -- multiple short cycles with fresh context:
```bash
bin/run-agent --loop 10 --max 5             # 10 cycles, 5 experiments each
bin/run-agent --loop 20 --tool gemini       # works with any AI CLI
```
Each cycle gets fresh context but reads accumulated file state (log.jsonl, TODO.md, findings docs) from previous cycles. If one cycle crashes, the next picks up from the file state. This is more resilient than a single long session because context doesn't degrade over time.

The loop stops automatically when:
- The hypothesis queue is empty
- The circuit breaker trips (5+ INVALID in last 20)
- The harness file was modified
- Max cycles reached

**Cron** -- scheduled recurring runs:
```bash
0 */6 * * * cd /path/to/SutroYaro && bin/run-agent --loop 3 --max 5 --researcher yad-cron
```

After any run:
```bash
bin/analyze-log          # text report
bin/analyze-log --plot   # generates results/progress.png
```

## The log.jsonl schema

Each line is one experiment. The schema is challenge-agnostic so it works for sparse parity now and nanoGPT later.

```json
{
  "id": "yad-034",
  "researcher": "yad",
  "date": "2026-03-11",
  "challenge": "sparse-parity",
  "hypothesis": "KM influence with 3 samples per bit instead of 5",
  "method": "km",
  "changed": {"influence_samples": 3},
  "baseline": {"method": "km", "influence_samples": 5, "ard": 1585},
  "result": {"ard": 2100, "dmc": 45000, "time_s": 0.004, "accuracy": 1.0},
  "delta_pct": {"ard": 32.5},
  "class": "LOSS",
  "notes": "3 samples gives noisy estimates, wrong bit selected 20% of runs"
}
```

Field definitions:

- `id`: `{researcher}-{sequence}`. Globally unique.
- `researcher`: who ran it (human or agent identity)
- `challenge`: which problem (`sparse-parity`, `nanogpt`, etc.)
- `method`: which algorithm family
- `changed`: dict of what was varied from baseline. One key = one variable.
- `baseline`: the comparison point (method + params + metrics)
- `result`: measured metrics from the locked harness
- `delta_pct`: percentage change from baseline per metric. Positive = worse for minimization targets (ARD, time), better for maximization targets (accuracy).
- `class`: one of `WIN` (improved primary metric), `LOSS` (worse), `INVALID` (couldn't run), `INCONCLUSIVE` (within noise), `BASELINE` (reference point)
- `notes`: free text, what was learned

The distinction between INVALID and LOSS matters. A crashed experiment is not a disproved hypothesis. INVALID means the experiment didn't produce valid measurements. LOSS means it ran correctly but the hypothesis was wrong.

## search_space.yaml

Defines the bounded mutation space per challenge. The agent can only vary parameters listed here. To add a new parameter or method, a human edits the file and commits.

This prevents agents from introducing unbounded variation (new architectures, code rewrites, external packages) without human review.

## Circuit breakers

Safety mechanisms borrowed from Tiny-Lab:

1. **INVALID rate**: if 5 of the last 20 log entries are INVALID, halt and require human review
2. **Lock file**: `research/.agent-lock` prevents concurrent agent cycles. Stale locks (>2 hours) are cleaned up.
3. **Max experiments per cycle**: default 20. The agent stops after this many, even if hypotheses remain.
4. **Harness checksum**: `bin/run-agent` verifies `src/harness.py` hasn't been modified before starting

## Moving to a new challenge

When the group moves from sparse parity to nanoGPT (or any new problem):

1. Add a new section to `research/search_space.yaml` with `challenge: nanogpt`
2. Add new questions to `research/questions.yaml`
3. Write a new harness function in `src/harness.py` (or a separate `src/harness_nanogpt.py`)
4. Establish baselines (logged as `class: BASELINE` entries in log.jsonl)
5. The rest of the infrastructure stays the same

See the "Adding nanoGPT" section at the bottom for a concrete proposal.

## How this differs from autoresearch

Karpathy's autoresearch (Mar 2026) uses a similar single-agent loop pattern: one mutable file, one locked harness, one metric, git as ratchet. Our design differs in:

| Aspect | autoresearch | SutroYaro |
|---|---|---|
| Researchers | 1 | Multiple, via PR |
| Scope | Optimize one training script | Compare methods across families |
| Log format | TSV (local only) | JSONL (mergeable across researchers) |
| Git strategy | Ratchet (branch = best, discard = reset) | History (all experiments kept) |
| Agent protocol | `program.md` (one file) | `AGENT.md` + `search_space.yaml` (structured) |
| Time budget | Fixed 5 min | Per-method (GF2 takes 500us, SGD takes 2min) |
| Challenge scope | Single (LLM pretraining) | Multi-challenge (sparse parity, then nanoGPT) |

We were already doing multi-researcher autonomous research before autoresearch dropped. This protocol formalizes what the group was already doing and adds machine-readable infrastructure.

## Adding nanoGPT (proposal)

Yaroslav's three-axis roadmap (Meeting #8) defines the final exam: energy-efficient training of Karpathy's nanoGPT. Sparse parity is practice.

### What changes

The core protocol stays identical. What changes per challenge:

**Harness**: `src/harness.py` gets a `measure_nanogpt()` function alongside `measure_sparse_parity()`. The nanoGPT harness measures:
- `val_bpb` (bits per byte on a fixed validation set, vocabulary-independent)
- `ard` (average reuse distance of one training step)
- `dmc` (data movement complexity)
- `time_s` (wall clock for fixed step count)
- `peak_memory_mb`

**Search space**: new block in `search_space.yaml`:

```yaml
challenge: nanogpt
version: 1

methods:
  - standard_sgd
  - adam
  - muon
  - sign_sgd
  - curriculum
  - per_layer

parameters:
  n_layers: [2, 4, 6, 8, 12]
  n_heads: [2, 4, 8]
  d_model: [64, 128, 256, 512]
  lr: [0.0001, 0.0003, 0.001, 0.003]
  batch_size: [4, 8, 16, 32]
  context_length: [64, 128, 256]
  weight_decay: [0.0, 0.01, 0.1]
  warmup_steps: [0, 100, 500]

metrics:
  primary: val_bpb
  secondary: [ard, dmc, time_s, peak_memory_mb]
  locked_in: src/harness.py
```

**Baselines**: run standard nanoGPT config, log as `class: BASELINE`, then the loop begins.

**Training budget**: fixed wall-clock (like autoresearch) makes sense here since model size varies. 5 minutes is reasonable for a tiny GPT on a single machine.

### What transfers from sparse parity

The sparse parity campaign produced knowledge that directly applies:

- **Per-layer updates work** without accuracy loss (may reduce ARD for deeper nets)
- **Curriculum learning transfers** (train small, expand)
- **W1 dominance** likely transfers (embedding matrix will dominate in nanoGPT)
- **Cache model matters more than raw ARD** (L2 can eliminate misses for small models)
- **Sign SGD converges faster** in some regimes (worth testing on nanoGPT)
- **Local learning rules fail on parity** but may work on language (worth re-testing)

These become starting hypotheses for the nanoGPT queue. The `research/questions.yaml` would include dependencies like:

```yaml
- id: nanogpt-Q0.1
  question: What is the baseline val_bpb for standard nanoGPT in 5 minutes?
  depends_on: []
  resolved_by: []

- id: nanogpt-Q1.1
  question: Does per-layer update reduce ARD on nanoGPT without hurting val_bpb?
  depends_on: [nanogpt-Q0.1]
  resolved_by: []
  prior: "Per-layer gave 3.8% ARD improvement on sparse parity (exp_a, exp_c)"
```

### Timeline

Moving to nanoGPT is a group decision. The protocol is ready whenever the group decides sparse parity has taught what it can teach. The three-axis roadmap suggests taking one step at a time: first improve process (this protocol), then improve metric (actual GPU measurement), then change problem (nanoGPT).

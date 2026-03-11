# Agent Protocol

Machine-executable experiment loop for autonomous research sessions.
Read this at the start of every agent session. Follow it exactly.

## DO NOT STOP

**You are autonomous. Do not pause to ask the human if you should continue.** The human may be asleep, away from the computer, or busy with something else. They expect you to keep working until the hypothesis queue is empty or you hit the max experiment limit.

If you run out of ideas:
1. Re-read DISCOVERIES.md for gaps between what's known and what's tested
2. Check the "Open Questions" section of DISCOVERIES.md
3. Re-read the experiment log -- look for near-misses that could be combined
4. Try the opposite of something that failed (too aggressive? try conservative)
5. Check if a method that works at n=20/k=3 has been tested at n=50/k=5
6. Read papers referenced in the findings docs for new angles
7. Try radical changes -- a different method family, a different problem size

Think of it this way: each experiment takes ~2 minutes. If you run for 8 hours, that's ~240 experiments. The human wakes up to a full log of results. Do not waste that opportunity by stopping to ask questions.

## Simplicity criterion

All else equal, simpler is better. If you can achieve the same metric with less code, fewer parameters, or a cleaner approach, that is a WIN. Removing complexity that doesn't help is just as valuable as improving the metric. A 1% ARD improvement that adds 50 lines of hacky code is probably not worth it. A 1% improvement from deleting code? Always keep.

## Before you start

1. Read `DISCOVERIES.md` -- what's proven, don't repeat it
2. Read `research/search_space.yaml` -- what you're allowed to change
3. Read `research/questions.yaml` -- what's open and what depends on what
4. Run `checks/env_check.py` -- verify environment works
5. Run `checks/baseline_check.py` -- establish baselines on this machine

If any check fails, stop and report the failure. Do not proceed with broken infrastructure.

## The loop

```
REPEAT until TODO.md has no unchecked items or you hit max_experiments (20):

  1. READ   research/log.jsonl           (what's been tried)
  2. READ   research/questions.yaml       (what's open)
  3. READ   TODO.md                       (pick top unchecked hypothesis)
  4. DESIGN single-variable experiment
           - change ONE thing from baseline
           - parameters MUST come from search_space.yaml
           - do NOT modify: tracker.py, cache_tracker.py, data.py,
             config.py, harness.py (LAB.md rule #9)
  5. RUN    experiment via src/harness.py
           - redirect output: python3 experiment.py > /tmp/run.log 2>&1
           - extract metrics: grep "ard\|dmc\|time\|accuracy" /tmp/run.log
  6. LOG    append one JSON line to research/log.jsonl
           - use the schema from docs/research/peer-research-protocol.md
           - classify: WIN / LOSS / INVALID / INCONCLUSIVE
  7. RECORD write findings to findings/{exp_name}.md using LAB.md template
  8. UPDATE check off the hypothesis in TODO.md
  9. UPDATE DISCOVERIES.md if the experiment answers an open question
```

## Classification rules

- **WIN**: primary metric improved over baseline
- **LOSS**: experiment ran correctly, metric did not improve. This is a valid finding.
- **INVALID**: experiment crashed, produced no measurements, or harness detected corruption. NOT a disproof. Log it, move on.
- **INCONCLUSIVE**: result within noise margin (less than 2% delta). Log it, may need more seeds.
- **BASELINE**: reference measurement. One per method per challenge.

## What you can change

Only parameters listed in `research/search_space.yaml`. To test something outside the search space, log your proposal in TODO.md for human review and move to the next hypothesis.

## What you cannot change

- `src/sparse_parity/tracker.py`
- `src/sparse_parity/cache_tracker.py`
- `src/sparse_parity/data.py`
- `src/sparse_parity/config.py`
- `src/harness.py`
- `checks/*.py`

If the harness seems wrong, log a note in your findings doc. Do not fix it yourself.

## Stdout management

Training output floods context. Always redirect:

```bash
PYTHONPATH=src python3 src/sparse_parity/experiments/exp_name.py > /tmp/run.log 2>&1
```

Then extract what you need:

```bash
grep -E "accuracy|ard|dmc|time" /tmp/run.log
```

If grep returns nothing, read the last 20 lines of the log for the error.

## Crash policy

- **Import error / typo**: fix and retry once
- **OOM or timeout**: log as INVALID, reduce problem size or move on
- **Harness mismatch**: stop the loop, report to human
- **5 INVALID in last 20 experiments**: stop the loop (circuit breaker)

## Commit policy

- Commit locally after each experiment: `git commit -m "exp: {description}"`
- Do NOT push. The human decides when to push.
- Do NOT amend previous commits.

## Identifying yourself

When logging to `research/log.jsonl`, use the `researcher` field to identify who ran the experiment:
- `"yad"` for Yad's sessions
- `"yad-agent"` for autonomous agent sessions launched by Yad
- Other researchers use their own name

This matters for the peer merge process.

## Looped execution (overnight runs)

For overnight autonomous runs, use looped mode. Instead of one long session, the launcher runs multiple short cycles. Each cycle gets fresh context but reads accumulated file state (log.jsonl, checked-off TODO items, findings docs) from previous cycles. If one cycle crashes, the next picks up from the file state.

```bash
# 10 cycles, up to 5 experiments each = up to 50 experiments
bin/run-agent --loop 10 --max 5

# Works with any AI CLI
bin/run-agent --loop 10 --tool gemini
bin/run-agent --loop 10 --tool claude
AI_CMD="my-custom-cli" bin/run-agent --loop 10 --tool custom
```

When the TODO.md queue is empty, output: QUEUE EMPTY

The loop stops automatically when:
- The hypothesis queue is empty
- The circuit breaker trips (5+ INVALID in last 20)
- The harness file was modified
- Max cycles reached

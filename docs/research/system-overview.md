# SutroYaro: system overview

A structured workspace where coding agents run energy-efficiency research. You point a coding agent at the repo, it reads the context files, runs experiments, and accumulates findings. Multiple people do this independently. The results merge through PRs.

This page describes how the pieces connect. Each piece has its own docs page. This is the map.

## What the system does

The Sutro Group wants to find training methods that use less energy. The toy problem is sparse parity (learn XOR from random bits). The real goal is nanoGPT. The workspace lets agents explore the method space, measure energy cost, and record what they find.

Three things happen in the workspace:

**Agents run experiments.** A coding agent reads CLAUDE.md and DISCOVERIES.md, picks a method, runs it against the locked harness, measures DMC (data movement cost), and writes up the finding. The two-phase protocol separates raw numbers (results.json) from interpretation (findings.md). LAB.md defines the rules. The run-experiment skill defines the steps.

**Agents get evaluated.** The eval environment (Gymnasium) wraps the same experiments into a benchmark. An agent picks from 16 methods, observes DMC, and gets graded on 12 categories (72 points). Did it find GF(2)? Did it notice local learning fails? Did it observe the ARD/DMC ranking disagreement? The answer key has 36 experiments as ground truth.

**Humans review and merge.** Contributors submit PRs. The reviewing agent checks locked files, findings format, log classification. If the result is significant, it gets a changelog entry. If not, it still merges. The weekly catch-up summarizes what happened across Telegram, Google Docs, and GitHub.

## How the pieces connect

```
CLAUDE.md ──────────────────────────── Agent reads this first
    │                                  (problem context, methods table,
    │                                   current best results)
    │
    ├── LAB.md ─────────────────────── Experiment protocol
    │     └── Two-phase results         Phase 1: results.json (numbers)
    │         Reproducibility rules     Phase 2: findings.md (analysis)
    │         Metric isolation          Don't edit tracker.py/harness.py
    │
    ├── DISCOVERIES.md ─────────────── What's proven (read before every experiment)
    │     └── 36 experiments            DMC rankings, failure modes, scaling walls
    │
    ├── AGENT.md ───────────────────── Autonomous loop protocol
    │     └── Pick hypothesis           From TODO.md or questions.yaml
    │         Run against harness       search_space.yaml bounds
    │         Classify result           WIN / PARTIAL / LOSS
    │         Log to log.jsonl          Append-only
    │
    ├── .claude/ ───────────────────── Claude Code specific
    │     ├── hooks/                    session-start (status), security-guard
    │     │                             (locked files), session-end (summary)
    │     ├── rules/                    Reproducibility, agent coordination
    │     ├── skills/                   run-experiment, weekly-catchup,
    │     │                             prepare-meeting
    │     └── settings.json             Hook configuration
    │
    ├── src/sparse_parity/eval/ ────── Eval environment
    │     ├── env.py                    Gymnasium: SparseParity-v0
    │     ├── grader.py                 12 categories, 72 points
    │     ├── registry.py               Add methods without editing env.py
    │     ├── backends.py               Local / Modal / Remote
    │     ├── answer_key.json           36 experiments as ground truth
    │     └── adapters/                 Anthropic, PrimeIntellect, HuggingFace
    │
    └── Automation ─────────────────── Sync and reporting
          ├── sync_telegram.ts          Pull Telegram messages
          ├── sync_google_docs.py       Pull Google Docs to markdown
          ├── docs/catchups/            Weekly summaries
          └── docs/sessions/            Video transcripts and chapters
```

## The two costs

Yaroslav's constraint: experiments should run within 1980s compute budgets. Under 1 second, ideally under 10ms.

The workspace measures two costs:

**Experiment cost.** How much data does the method move? DMC (data movement complexity) captures this. GF(2) at DMC 8,607 is 9 million times cheaper than Fourier at 78 billion. 4 of 16 methods run under 10ms. The eval environment tracks this per step.

**Agent cost.** How many experiments did the agent need? How many tokens of reasoning? The eval environment's efficiency category awards 5 points for finding the best method in 3 steps. But we don't yet track token cost or the agent's thinking time between steps. The goal is to measure both: can an agent find GF(2) using few tokens and small wall clock time, without trying all 16 methods?

## Who does what

**The coding agent** reads CLAUDE.md, runs experiments, writes findings. It doesn't need to know about the eval environment or the hooks. Those are layers on top.

**The hooks** (Claude Code only) fire automatically. Session-start shows status so the agent doesn't ask "catch me up." The security guard blocks edits to measurement code. Session-end shows what changed. Other coding agents (Gemini, Codex) skip hooks and read CLAUDE.md directly.

**The eval environment** is a different mode. Instead of the agent writing experiment code, it picks from 16 existing methods and observes results. The grading checks whether it made specific discoveries. This can run locally, through Anthropic tool calls, on PrimeIntellect, or as a HuggingFace leaderboard.

**The weekly catch-up** syncs Telegram, Google Docs, and GitHub into a summary page. The prepare-meeting skill compiles recent experiments into a report.

**Contributors** fork the repo, run experiments, submit PRs. The reviewing agent checks the PR against the experiment protocol. See the [changelog](../changelog.md) for merged contributions.

## What each file is for

| File | Who reads it | What it does |
|------|-------------|-------------|
| CLAUDE.md | All agents | Problem context, methods table, best results |
| LAB.md | Agents running experiments | Protocol, rules, templates |
| AGENT.md | Autonomous agent loop | Pick hypothesis, run, classify, log |
| AGENT_EVAL.md | Agents using the eval env | How to add methods, run evals, read grading |
| DISCOVERIES.md | Everyone, before every experiment | Proven facts, open questions |
| TODO.md | Agents looking for work | Hypothesis queue |
| .claude/settings.json | Claude Code | Hook configuration |
| .claude/rules/*.md | Claude Code | Reproducibility, coordination constraints |
| .claude/skills/*/SKILL.md | Claude Code | Workflow definitions |

## What still needs to happen

This workspace handles sparse parity. The path to nanoGPT requires:

1. Actual GPU energy measurement (Issue #6: compare DMC vs ARD vs real joules on an H100)
2. A harder problem (nanoGPT character-level training, Issue #9)
3. Agent cost tracking (tokens per discovery, thinking time between steps)
4. PR review automation (Issue #54: webhook + Claude Code + Telegram approval)

The workspace itself is the product. Improving how agents find better algorithms is progress along Yaroslav's "process" axis, independent of which problem they're solving.

## Docs map

| Page | What it covers |
|------|---------------|
| [Eval environment](eval-environment.md) | Gymnasium env, grading, adapters, baselines |
| [Agent infrastructure](agent-infrastructure.md) | Hooks, rules, skills, V2 diagram |
| [Adding a challenge](adding-a-challenge.md) | Step-by-step for new problems |
| [Adding an eval challenge](adding-an-eval-challenge.md) | Registry, answer key, baselines |
| [Survey](survey.md) | All 33 experiments ranked |
| [Context](../context.md) | Group history, timeline, people |
| [Peer research protocol](peer-research-protocol.md) | Multi-researcher workflow |

# Tooling

How we use Claude Code as a research automation tool for the Sutro Group.

## How It Works

![Agent Workflow](../diagrams/agent-workflow.png)

The human writes specs (CLAUDE.md, DISCOVERIES.md, LAB.md). The lead agent reads those, surveys the problem space, then dispatches isolated sub-agents in parallel. Each sub-agent gets one approach, the experiment template, and shared modules. No sub-agent sees another's results. Outputs feed back into DISCOVERIES.md for the next round.

## The Stack

| Tool | What it does |
|------|-------------|
| [Claude Code](claude-code-setup.md) | AI coding agent in the terminal — runs experiments, writes findings, manages the MkDocs site |
| [CLAUDE.md](claude-code-setup.md#claudemd) | Project instructions file that gives Claude context about the repo |
| [Superpowers plugin](skills.md) | [obra/superpowers](https://github.com/obra/superpowers) — brainstorming, TDD, debugging, parallel agents, code review |
| [Custom skills](skills.md#custom-skills) | Anti-slop guide, Ralph Wiggum |
| [MCP servers](skills.md#mcp-servers) | Extensible tool servers (Google Docs, browser, diagrams) |
| [Anti-slop guide](anti-slop-guide.md) | Reference for eliminating AI writing patterns from prose |
| [Automation scripts](automation.md) | `sync_google_docs.py` for pulling Google Docs, `sync_telegram.ts` for pulling Telegram threads, session trace export |
| [Sync runbook](sync-runbook.md) | Weekly/daily/per-session checklists for keeping the knowledge base current |

## Scripts (`bin/`)

| Script | What it does | Example |
|--------|-------------|---------|
| `bin/run-agent` | Autonomous experiment loop (any AI CLI) | `bin/run-agent --tool claude --max 10` |
| `bin/merge-findings` | Import contributor results via PR | `bin/merge-findings path/to/log.jsonl` |
| `bin/analyze-log` | Progress report from experiment log | `bin/analyze-log --plot` |
| `bin/reproduce-all` | Verify all 14 experiments in 0.28s | `PYTHONPATH=src python3 bin/reproduce-all --budget 10` |
| `bin/gpu_energy.py` | Real GPU energy via Modal Labs (L4) | `modal run bin/gpu_energy.py` |

### GPU Energy Measurement

`bin/gpu_energy.py` runs harness methods on an NVIDIA L4 via [Modal Labs](https://modal.com) and samples actual power draw (watts) via pynvml during execution. Reports joules per method alongside ARD/DMC proxy metrics.

```bash
pip install modal
modal token set          # authenticate (one-time)
modal run bin/gpu_energy.py          # run all defaults
modal run bin/gpu_energy.py --json   # machine-readable
```

Cost: under $0.01 per run. See [GPU energy baseline](../findings/gpu_energy_baseline.md) for the first results.

### Reproduce All

`bin/reproduce-all` runs every method across all 3 challenges and verifies results match baselines. Supports a `--budget` flag (ms) to skip experiments over a time ceiling.

```bash
PYTHONPATH=src python3 bin/reproduce-all             # all 14 in 0.28s
PYTHONPATH=src python3 bin/reproduce-all --budget 10  # 6 pass in 0.08s
```

## What Worked

The combination that produced 16 experiments in a few days:

1. **CLAUDE.md as shared context** — Every Claude Code session starts by reading the project state, findings, and working style rules
2. **LAB.md as experiment protocol** — Enforces one-hypothesis-per-experiment, baselines, and commit discipline
3. **Anti-slop on all prose** — Keeps documentation readable by humans, not just LLMs
4. **Parallel agents** — Multiple Claude Code instances running independent experiments simultaneously
5. **Sub-2-second iteration** — `fast.py` (numpy) keeps the feedback loop tight enough for hundreds of experiments per hour

## What to Try Next

- MCP servers for direct Google Docs access (currently using export URLs)
- Hooks for auto-running experiments on file save
- Custom skills for the Sutro research loop (literature search, hypothesis, experiment, measure)
- Memory files for cross-session learning about what hyperparameters work

# Tooling

How we use Claude Code as a research automation tool for the Sutro Group.

## How It Works

```mermaid
flowchart TD
    H["Human (You)"] -->|writes specs| Docs
    H -->|prompts| Lead

    subgraph Repo["GitHub Repo (SutroYaro)"]
        Docs["CLAUDE.md · DISCOVERIES.md · LAB.md\nproposed-approaches.md"]
        Scripts["sync_telegram.ts · sync_google_docs.py\nexport_sessions.py"]
    end

    Lead["Lead Agent\n(Claude Code)"] -->|reads first| Docs
    Scripts -->|feeds context| Lead

    Lead -->|1. reads docs| Step1["Survey problem space\n(27 approaches)"]
    Step1 -->|2. dispatches| Agents

    subgraph Agents["Isolated Sub-Agents (x17, parallel)"]
        A1["GF(2)"]
        A2["Hebbian"]
        A3["RL Bandit"]
        A4["..."]
    end

    Agents -->|each produces| Out["experiment.py · results.json · findings.md"]
    Out -->|feeds back into| Docs

    style H fill:#fff2cc,stroke:#d6b656,color:#333
    style Repo fill:#f0f4ff,stroke:#6c8ebf,color:#333
    style Docs fill:#e1d5e7,stroke:#9673a6,color:#333
    style Scripts fill:#d5e8d4,stroke:#82b366,color:#333
    style Lead fill:#ffe6cc,stroke:#d79b00,color:#333
    style Step1 fill:#fff2cc,stroke:#d6b656,color:#333
    style Agents fill:#fce4ec,stroke:#b85450,color:#333
    style A1 fill:#f8cecc,stroke:#b85450,color:#333
    style A2 fill:#f8cecc,stroke:#b85450,color:#333
    style A3 fill:#f8cecc,stroke:#b85450,color:#333
    style A4 fill:#f8cecc,stroke:#b85450,color:#333
    style Out fill:#d5e8d4,stroke:#82b366,color:#333
```

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

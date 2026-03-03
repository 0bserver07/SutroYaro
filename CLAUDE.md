# CLAUDE.md - Sutro Group Research Workspace

## Project Context

This is Yaroslav's research workspace for the **Sutro Group**, a study group exploring energy-efficient AI training. The group meets weekly at South Park Commons in San Francisco.

## Core Concepts

- **Sparse Parity**: The benchmark task -- learn XOR/parity from random positive/negative numbers, scaling to 20 bits with only 3 relevant
- **Average Reuse Distance (ARD)**: Proxy metric for energy efficiency. Small ARD = data stays in cache = cheap. Large ARD = expensive external memory access
- **Energy-efficient training**: The overarching goal -- reduce joules-per-training-step, primarily by reducing memory access costs
- **Forward-Forward algorithm**: Hinton's alternative to backprop, discussed as a candidate for better ARD

## Key People

- **Yaroslav** (repo owner) - Technical sprints, algorithm work
- **Emmett** - Aster agentic loop framework, 2x energy improvement on microgpt
- **Germaine**, **Andy**, **Seth**, **Barak**, **Jamie Simon** - Group members

## Source Materials

The `docs/google-docs/` directory contains markdown conversions of the primary Google Docs:
- `challenge-1-sparse-parity.md` - The current challenge definition
- `sutro-group-main.md` - Meeting index with links to all notes
- `yaroslav-technical-sprint-1.md` - Detailed sprint log (02 Mar 2026)

## Related Repos

- https://github.com/cybertronai/sutro - Main code repo with sparse_parity_benchmark.py
- https://github.com/0bserver07/SutroYaro - This research workspace

## Working Style

- Iteration time must stay under 1 second
- Change one thing at a time (correctness, then speed, then energy)
- Priority: correctness > wall-clock time > energy usage
- Checkpoint every correct+fast solution

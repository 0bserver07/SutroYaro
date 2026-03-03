# SutroYaro

Yaroslav's research workspace for the Sutro Group -- a study group focused on energy-efficient AI training.

## About

The Sutro Group meets weekly at South Park Commons (SF) to explore reinventing AI learning algorithms from first principles with energy efficiency as a core constraint. The key insight: **memory cost is the biggest contributor to energy use** in neural network training.

## Current Focus: Sparse Parity Challenge

The group's current challenge uses sparse parity as a "drosophila" of energy-efficient training:
- Start with XOR/parity as the simplest non-trivial learning task
- Scale to 3-bit parity with 17 noise bits (20 total)
- Measure energy via **Average Reuse Distance** (ARD) as a proxy metric
- Use AI agents to suggest algorithmic improvements

## Structure

```
docs/
  google-docs/       # Converted Google Docs (source of truth)
  meeting-notes/     # Per-meeting detailed notes
  lectures/          # Lecture materials, papers, presentations
  homework/          # Homework assignments and submissions
research/            # Deep research notes, literature review
findings/            # Key findings and results
plans/               # Technical sprint plans, roadmaps
src/                 # Code, experiments, notebooks
```

## Key Results (Sprint 1)

- Gradient fusion strategy identified automatically by Claude
- Cache reuse improved by 16%
- Conclusion: significant gains require fundamentally different algorithms (forward-forward, per-layer update, etc.)
- Candidate algorithms deferred to Technical Sprint 2

## Links

- Telegram: https://t.me/sutro_group
- GitHub (cybertronai/sutro): https://github.com/cybertronai/sutro
- Meetings: Mondays 18:00 at South Park Commons (380 Brannan St)

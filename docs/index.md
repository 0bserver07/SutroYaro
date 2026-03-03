# SutroYaro

Yaroslav's research workspace for the **Sutro Group** -- a study group focused on energy-efficient AI training.

## About

The Sutro Group meets weekly at South Park Commons (SF) to explore reinventing AI learning algorithms from first principles with energy efficiency as a core constraint.

!!! info "Key Insight"
    **Memory cost is the biggest contributor to energy use** in neural network training. Local registers cost ~5pJ vs HBM at ~640pJ -- a 128x difference.

## Current Focus: Sparse Parity Challenge

The group's current challenge uses sparse parity as a "drosophila" of energy-efficient training:

- Start with XOR/parity as the simplest non-trivial learning task
- Scale to 3-bit parity with 17 noise bits (20 total)
- Measure energy via **Average Reuse Distance** (ARD) as a proxy metric
- Use AI agents to suggest algorithmic improvements

## Quick Links

| Resource | Link |
|----------|------|
| Telegram | [t.me/sutro_group](https://t.me/sutro_group) |
| Code repo | [cybertronai/sutro](https://github.com/cybertronai/sutro) |
| Meetings | Mondays 18:00 at South Park Commons (380 Brannan St) |
| Bill Daly talk | [Energy use in GPUs](https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457) |

## Sprint 1 Results

- Gradient fusion strategy identified automatically by Claude
- Cache reuse improved by **16%**
- Conclusion: significant gains require fundamentally different algorithms
- Next: Forward-Forward, per-layer updates, local learning rules

## Repo Structure

```
docs/                # All documentation (mkdocs source)
  google-docs/       # Converted Google Docs
  meetings/          # Meeting notes index
  homework/          # Homework assignments
  lectures/          # Lecture materials
  findings/          # Key findings
  plans/             # Sprint plans
  research/          # Deep research notes
src/                 # Scripts and tools
  sync_google_docs.py  # Standalone Google Docs sync
```

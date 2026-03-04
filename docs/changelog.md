# Changelog

All notable changes to this research workspace.

## [Unreleased]

### Planned
- Sign SGD (Kou et al. 2024) — can it solve k=5?
- Cache-aware MemTracker — simulate L1/L2 hits for realistic energy numbers
- Curriculum learning — train on small n, transfer to large n
- Weight decay sweep for faster grokking

---

## [0.6.0] - 2026-03-04

### Research experiments (Tasks A-F)
- **Exp A: ARD on winning config** — per-layer gives 3.8% ARD improvement on 20-bit (17,299 vs 17,976). W1 dominates at 75% of all float reads, capping reorder-based improvements at ~10%.
- **Exp B: Batch ARD** — batch-32 does 16x fewer parameter writes but shows 17x higher ARD in our metric. The metric doesn't model cache. On real hardware where W1 fits in L2, batch training would be far more efficient. Need cache simulation in MemTracker.
- **Exp C: Per-layer on 20-bit** — per-layer forward-backward converges identically to standard backprop (99.5%, same epoch count). Free 3.8% ARD improvement with zero accuracy cost.
- **Exp D: Scaling frontier** — standard SGD breaks at n^k > 100,000 iterations. k=3 works up to n~30-45. k=5 is categorically impractical (~200,000 epochs for n=20). This is where novel algorithms matter.
- **Exp E: Forward-Forward** — solves 3-bit (100%) but fails 20-bit (58.5%). Has 25x WORSE ARD than backprop. The "local learning" advantage is illusory for 2-layer networks.
- **Exp F: Prompting strategies** — documented the literature-search → diagnose → experiment workflow that took us from 54% to 100%.

### Speed
- **`fast.py`**: numpy-accelerated training solves 20-bit in 0.12s average (220x faster than pure Python). hidden=200, n_train=1000, batch=32.

### Infrastructure
- LAB.md — protocol for autonomous Claude Code experiment sessions
- DISCOVERIES.md — accumulated knowledge base from all experiments
- `_template.py` — copy-and-modify experiment starter

---

## [0.5.0] - 2026-03-04

### Added
- **Exp 1: Fix Hyperparams** — 99% accuracy on 20-bit sparse parity (k=3). Changed LR 0.5→0.1, batch_size 1→32, n_train 200→500. Classic grokking: stuck at 50% for 40 epochs, then phase transition to 99% in ~10 epochs.
- **Exp 4: GrokFast** — counterproductive. Baseline SGD hits 100% in 5 epochs (22.7s). GrokFast took 12 epochs and never reached 100%. The bottleneck was wrong hyperparams, not the optimizer.
- Literature review: 6 key papers (Barak 2022, Kou 2024, Merrill 2023, GrokFast, NTK grokking, SLT phase transitions)
- Research plan for autonomous experiment cycle
- Results organized into per-run directories with auto-generated index

---

## [0.4.0] - 2026-03-03

### Added
- Complete sparse parity pipeline (all 5 phases)
    - 3 training variants: standard backprop, fused, per-layer
    - MemTracker for ARD measurement
    - JSON + markdown + plot output
- 20/20 tests passing
- Per-run results directories with index lookup

---

## [0.3.0] - 2026-03-03

### Added
- MkDocs site with Material theme
- Mermaid diagrams in context and findings
- Changelog tracking

---

## [0.2.0] - 2026-03-03

### Added
- `src/sync_google_docs.py` -- standalone script to sync Google Docs to markdown
- Auto-extracted references (`docs/references_auto.md`)
- Expanded homework archive covering all 7 meetings

---

## [0.1.0] - 2026-03-03

### Added
- Initial research environment setup
- Converted 3 Google Docs to local markdown:
    - Challenge #1: Sparse Parity (spec)
    - Sutro Group Main (meeting index)
    - Yaroslav Technical Sprint 1 (sprint log)
- Extracted 30+ hyperlinks into `docs/references.md`
- Directory structure: docs, findings, plans, research, src
- `CLAUDE.md` with project context for AI assistants
- `CONTEXT.md` with project background and timeline
- Sprint 1 findings documented
- Sprint 2 plan drafted
- Meeting notes index with all external links
- Lectures and homework directories
- `.gitignore` for Python/Jupyter/IDE files

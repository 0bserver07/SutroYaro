# Task 4: Reproduce Germain's depth-1/hidden-64 ARD result

**Priority**: MEDIUM
**Status**: DONE
**Source**: Yaroslav verification doc, Meeting #8

## Context

Germain's supervisor/researcher harness found:

- Baseline ARD: ~48.1-48.9
- Depth-1 + Hidden-64 ARD: ~33.1-34.5
- Accuracy gate passed (>=0.90), time under 2s

But Yaroslav flagged: "the winning change coincides with a big drop in total_accesses (92k to 64k). That might be legitimately less work (fewer layers/params/state)... more research needed."

We need to check: is this a real energy improvement, or just doing less computation? If total work drops proportionally to the ARD drop, it's not a real locality win.

## Tasks

- [ ] Run our baseline with depth=1, hidden=64 using fast.py
- [ ] Measure ARD, total_accesses, accuracy, wall time
- [ ] Compare ARD per unit of useful work (normalize by total_accesses or by accuracy)
- [ ] Check if it solves n=20/k=3 reliably (Germain's was on a simpler config?)
- [ ] Write findings

## Results (5 seeds, n=20/k=3)

| Config | Accuracy | ARD | DMC | Total Floats |
|--------|----------|-----|-----|-------------|
| hidden=200 (baseline) | 100% | 6,589 | 740,165 | 18,308 |
| hidden=64 (Germain) | 100% | 2,129 | 135,724 | 5,904 |
| **Improvement** | - | **67.7%** | **81.7%** | **67.7%** |

**Verdict**: Yaroslav was right to be skeptical. ARD/float is identical (0.360 vs 0.361). The ARD improvement is entirely explained by the model being smaller (fewer parameters = fewer floats to access). The locality per unit of work is unchanged. Hidden=64 is not a locality win, it's just doing less computation. Both solve n=20/k=3 at 100%.

This is still useful information: for sparse parity, hidden=64 is sufficient and uses 3x less energy. But it's not transferable to harder problems where you need more capacity.

## References

- Yaroslav verification: docs/google-docs/yaroslav-verification.md (Germain section)
- Our baseline: src/sparse_parity/fast.py (hidden=200, 2 layers)
- G B's prior result in CLAUDE.md: "Architecture experiments (depth-1/hidden-64, ARD ~33-35)"

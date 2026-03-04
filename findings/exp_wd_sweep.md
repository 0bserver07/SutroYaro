# Experiment WD_SWEEP: Weight Decay Sweep

**Date**: 2026-03-04
**Status**: SUCCESS
**Answers**: Open Question #5 — "Does higher WD (0.1, 1.0) accelerate grokking on 20-bit?"

## Hypothesis

If we increase weight decay beyond 0.01, grokking will accelerate because stronger regularization encourages simpler (sparser) solutions faster.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20 |
| k_sparse | 3 |
| hidden | 200 |
| lr | 0.1 |
| wd | sweep: 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0 |
| batch_size | 32 |
| max_epochs | 200 |
| n_train | 1000 |
| seeds | 42-46 (5 seeds per WD) |
| method | standard (numpy SGD, hinge loss) |

## Results

| Metric | Value |
|--------|-------|
| Best WD | 0.01 (39 avg epochs, 100% success) |
| Runner-up WD | 0.05 (45.8 avg epochs, 100% success) |
| Working range | [0.01, 0.05] only |
| Failure modes | WD<0.01: no convergence in 200 epochs; WD>=0.1: regularization kills learning |

## Key Table

| WD | Avg Epochs | Avg Time (s) | Success Rate |
|---------|------------|--------------|--------------|
| 0.001 | FAIL | 0.314 | 0% |
| **0.01** | **39.0** | **0.108** | **100%** |
| 0.05 | 45.8 | 0.124 | 100% |
| 0.1 | FAIL | 1.124 | 0% |
| 0.5 | FAIL | 0.834 | 0% |
| 1.0 | FAIL | 0.660 | 0% |
| 2.0 | FAIL | 0.744 | 0% |

## Analysis

### What worked
- WD=0.01 remains the best setting — 100% success, fastest average (39 epochs, 0.108s)
- WD=0.05 also works perfectly — slightly slower (45.8 epochs) but still robust
- The working range [0.01, 0.05] is narrow but reliable

### What didn't work
- WD=0.001 (too weak): weights grow unconstrained, no phase transition in 200 epochs
- WD>=0.1 (too strong): regularization penalty dominates the loss, weights get shrunk too aggressively for the network to learn the parity function
- No WD value beats the existing default of 0.01

### Surprise
- The working WD range is extremely narrow — only a 5x range (0.01-0.05) out of a 2000x sweep. This suggests WD is tightly coupled to LR: the effective regularization is LR*WD, so the working range for LR*WD is [0.001, 0.005]. WD=0.001 fails because LR*WD=0.0001 is too weak; WD=0.1 fails because LR*WD=0.01 is too strong.
- WD=0.001 failing is notable — without enough regularization, the phase transition doesn't happen at all within 200 epochs. Weight decay isn't just a nice-to-have; it's essential for grokking in this regime.

## Open Questions (for next experiment)

- Does the working WD range shift with LR? e.g., LR=0.05 might work with WD=0.1 (keeping LR*WD=0.005)
- Does WD=0.001 eventually solve if given more epochs (500+)? Would confirm WD controls grokking speed, not capability
- Can a WD schedule (start high, decay) accelerate the phase transition?

## Files

- Experiment: `src/sparse_parity/experiments/exp_wd_sweep.py`
- Results: `results/exp_wd_sweep/results.json`

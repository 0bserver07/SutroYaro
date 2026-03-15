# Experiment: Which Proxy Predicts GPU Energy?

**Date**: 2026-03-14
**Status**: FINDING
**Issue**: #6

## Question

Yaroslav said ARD is too coarse as an energy proxy. Is DMC better? Or is something else entirely the right metric at this scale?

## Setup

Ran 8 experiments on an NVIDIA L4 via Modal Labs (`bin/gpu_energy.py`). Measured real watts via pynvml during execution. Compared against ARD and DMC from the local harness.

Methods too fast for the power sampler (under 5ms) were excluded from the correlation analysis because pynvml couldn't get a reading before they finished.

## Data

| Challenge | Method | Joules | ARD | DMC | Time (ms) |
|-----------|--------|--------|-----|-----|-----------|
| parity | sgd | 8.601 | 8,504 | 1,278,460 | 533 |
| parity | fourier | 0.878 | 11,980,500 | 78,140,662,852 | 54 |
| sum | sgd | 0.033 | 20 | 2,862 | 2.0 |
| sum | km | 0.051 | 92 | 20,632 | 3.2 |
| and | sgd | 0.291 | 29,164 | 52,885,890 | 18 |
| and | km | 0.046 | 92 | 165,060 | 2.9 |

GF(2) and parity-KM finished in under 5ms, before the power sampler could take a reading.

## Correlation

| Predictor | Spearman r | p-value |
|-----------|-----------|---------|
| Wall-clock time | 1.000 | 0.000 |
| ARD | 0.812 | 0.050 |
| DMC | 0.771 | 0.072 |

Log-space Pearson (linear fit in log-log):

| Predictor | Pearson r (log) |
|-----------|----------------|
| log(time) | 1.000 |
| log(ARD) | 0.665 |
| log(DMC) | 0.548 |

## Finding

**Wall-clock time perfectly predicts GPU energy at this scale.** The L4 draws near-constant power (16.1W) regardless of which method runs. Joules = 16.1 * seconds. The GPU is barely loaded by these workloads.

ARD ranks methods in roughly the right order (r=0.812) but breaks on Fourier. Fourier has the worst ARD (11.9M) and worst DMC (78B) of any method, but uses less energy than SGD (0.878J vs 8.601J). The reason: Fourier streams through data with no reuse, which ARD penalizes heavily, but finishes fast because the operations are simple. ARD measures memory access patterns. Energy on an idle GPU measures time.

DMC is slightly worse than ARD (r=0.771 vs 0.812), not better. The sqrt(stack_distance) weighting doesn't help at this scale.

## What this means

For sparse parity on a single GPU, use wall-clock time as the energy proxy. It's perfect and free.

ARD and DMC matter when the workload is large enough to cause variable power draw (cache misses, HBM bandwidth saturation, compute unit utilization). That happens at nanoGPT scale, not at n=20/k=3 scale. When the group moves to larger problems, revisit this comparison.

For now: keep ARD as a secondary metric for understanding algorithm behavior (memory access patterns), but optimize for time.

## How to reproduce

```bash
# Local proxy metrics
PYTHONPATH=src python3 bin/reproduce-all

# GPU energy (requires Modal account)
modal run bin/gpu_energy.py
```

## Files

- Script: `bin/gpu_energy.py`
- This document: `findings/exp_proxy_comparison.md`
- GPU energy baseline: `findings/gpu_energy_baseline.md`
- Reproduce: `modal run bin/gpu_energy.py` (requires Modal account)

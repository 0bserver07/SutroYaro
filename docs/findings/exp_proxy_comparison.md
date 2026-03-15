# Experiment: GPU vs CPU for Sparse Parity Methods

**Date**: 2026-03-14
**Status**: INCONCLUSIVE (proxy question) / FINDING (GPU overhead)
**Issue**: #6

## Question

Sub-question of "Is ARD or DMC the better energy proxy?" (Issue #6). Before comparing proxies, we needed to know: does GPU measurement produce useful energy data for sparse parity?

## What was performed

Reimplemented three sparse parity methods (GF(2), SGD, KM) in PyTorch CUDA and ran them on an NVIDIA L4 via Modal Labs. Matched the numpy harness config: n=20, k=3, hidden=200, lr=0.1, wd=0.01, batch=32, hinge loss, seed=42.

Earlier attempt used numpy on a GPU container (CPU-bound, GPU idle, data useless). This version uses PyTorch with tensors on CUDA so the GPU does the actual compute.

## What was produced

| Method | Acc | GPU time | CPU time (numpy) | GPU/CPU ratio |
|--------|-----|----------|-------------------|---------------|
| GF(2) | 1.00 | 1.7ms | 0.5ms | 3.4x slower |
| SGD | 1.00 | 1014ms (37 epochs) | 142ms (40 epochs) | 7.1x slower |
| KM | 1.00 | 663ms | 1.1ms | 603x slower |

GPU drew 15.9W (near idle). Cost: $0.002 per run.

## Can it be reproduced?

```bash
pip install modal
modal token set
modal run bin/gpu_energy.py
```

Requires Modal account. Costs under $0.01. The image builds PyTorch on first run (~90s), cached after that.

## Finding

**Sparse parity at n=20/k=3 is too small for GPU.** All three methods are slower on GPU than CPU:

- GF(2) is sequential row reduction (XOR). CUDA can't parallelize it. Runs on CPU even when called from a GPU container.
- SGD has the same epoch count (37 vs 40) but each epoch is slower because the weight matrix (200x20) is too small for CUDA kernel launch overhead to amortize.
- KM is 603x slower because it does 20 independent small operations. Each one launches a CUDA kernel for a tiny tensor. The overhead dominates.

The GPU drew near-idle power (15.9W) because the compute units were barely used. Energy measurement via pynvml on the earlier run showed constant wattage across all methods, confirming the workloads don't stress the hardware.

**The proxy comparison (ARD vs DMC vs joules) remains unanswered.** It requires workloads large enough to produce variable GPU power draw. At sparse parity scale, the GPU adds overhead and the power reading is just idle draw times wall clock.

**What this means for the project:** GPU energy measurement becomes useful at nanoGPT scale. For sparse parity, CPU wall-clock time is the most honest performance metric. The `bin/gpu_energy.py` pipeline works and is ready for larger workloads.

Yaroslav's direction of verifying picojoule numbers for register vs cache vs HBM operations requires CUDA/PTX kernels targeting specific memory tiers, not running Python methods on a GPU.

## Files

- Script: `bin/gpu_energy.py`
- This document: `findings/exp_proxy_comparison.md`
- Reproduce: `modal run bin/gpu_energy.py`

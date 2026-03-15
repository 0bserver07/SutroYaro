# Experiment: GPU vs CPU for Sparse Parity Methods

**Date**: 2026-03-14
**Status**: FINDING
**Issue**: #6

## Question

Does running sparse parity methods on GPU (PyTorch CUDA) produce useful energy or performance data? Sub-question of "Is ARD or DMC the better energy proxy?"

## What was performed

Reimplemented three sparse parity methods (GF(2), SGD, KM) in PyTorch CUDA. Ran them on an NVIDIA L4 via Modal Labs, matching the numpy harness config: n=20, k=3, hidden=200, lr=0.1, wd=0.01, batch=32, hinge loss, seed=42.

Ran 5 times to measure variance. Compared against CPU numpy baselines from `bin/reproduce-all`.

## What was produced

### GPU times (5 runs on NVIDIA L4 via Modal)

| Run | GF(2) | SGD (37 epochs) | KM |
|-----|-------|------------------|-----|
| 1 | 1.7ms | 1014ms | 663ms |
| 2 | 2.1ms | 1676ms | 1016ms |
| 3 | 2.0ms | 1367ms | 844ms |
| 4 | 2.3ms | 1603ms | 899ms |
| 5 | 2.0ms | 1571ms | 921ms |
| **Mean** | **2.0ms** | **1446ms** | **869ms** |
| **Std** | **0.2ms** | **254ms** | **127ms** |

100% accuracy on all 5 runs, 37 epochs for SGD on all 5 runs.

### GPU vs CPU comparison

| Method | CPU (numpy) | GPU mean | GPU/CPU ratio |
|--------|-------------|----------|---------------|
| GF(2) | 0.5ms | 2.0ms | 4x slower |
| SGD | 142ms | 1446ms | 10x slower |
| KM | 1.1ms | 869ms | 790x slower |

### Cost

$0.002-0.003 per run. Total for 5 runs: ~$0.012.

## Can it be reproduced?

```bash
# GPU (requires Modal account)
pip install modal
modal token set
modal run bin/gpu_energy.py

# CPU baseline
PYTHONPATH=src python3 bin/reproduce-all
```

## Finding

**GPU is 4-790x slower than CPU for sparse parity at n=20/k=3.** Consistent across 5 runs.

- GF(2) is sequential row reduction (XOR). Can't parallelize. Runs on CPU even inside a GPU container. 4x overhead from container/PyTorch setup.
- SGD has the same epoch count (37) but each epoch is 10x slower. The weight matrix (200x20) is too small for CUDA kernel launch overhead to amortize.
- KM is 790x slower. It runs 20 small independent operations, each launching a CUDA kernel for a tiny tensor.

**The ARD vs DMC proxy comparison is still unanswered.** These workloads don't stress the GPU memory subsystem, so measuring GPU energy here tells you nothing about memory access patterns. That question needs nanoGPT-scale workloads.

**What's useful:** The `bin/gpu_energy.py` pipeline works (PyTorch on Modal, matching Yaroslav's gpu_toy.py pattern). When the group moves to nanoGPT, this script is the starting point. At sparse parity scale, use CPU wall-clock time.

## Files

- Script: `bin/gpu_energy.py`
- This document: `findings/exp_proxy_comparison.md`
- Reproduce: `modal run bin/gpu_energy.py`

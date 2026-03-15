# Experiment: Sparse Sum vs Sparse Parity (All Methods, CPU and GPU)

**Date**: 2026-03-15
**Status**: FINDING
**Related**: Issue #4, Issue #5

## Question

Yaroslav asked: redo all experiments for sparse sum. How does every method compare between sum and parity, on both CPU and GPU?

## What was performed

Ran all available methods on both tasks (sparse parity and sparse sum) at n=20, k=3, seed=42.

- CPU: numpy harness (`bin/reproduce-all`), verified across 5 seeds
- GPU: PyTorch CUDA on NVIDIA L4 via Modal (`bin/gpu_energy.py`), 4 runs each

## What was produced

### CPU (numpy, 5-seed mean)

| Method | Parity time | Parity acc | Sum time | Sum acc |
|--------|-------------|------------|----------|---------|
| GF(2) | 0.5ms | 100% | fails | 0% |
| KM | 1.1ms | 100% | 0.7ms | 100% |
| OLS | n/a | n/a | 0.5ms | 100% |
| Fourier | 12ms | 100% | 0.3ms | 100% |
| SGD (harness) | 142ms (40 ep) | 100% | 0.8ms (1 ep) | 100% |
| Sign SGD | 29ms mean (unreliable) | 3-5/5 | 3.7ms | 5/5 |

### GPU (PyTorch CUDA, L4, 4-run mean)

| Method | Parity time (5 runs) | Sum time (4 runs) |
|--------|---------------------|-------------------|
| GF(2) | 2.0ms (CPU, can't parallelize) | 2.7ms (fails on sum) |
| SGD | 1446ms, 37 epochs | 452ms, 1 epoch |
| KM | 869ms | 197ms |

### GPU vs CPU ratio

| Method | Parity GPU/CPU | Sum GPU/CPU |
|--------|---------------|-------------|
| SGD | 10x slower | 565x slower |
| KM | 790x slower | 281x slower |

## Can it be reproduced?

```bash
# CPU (all methods, all challenges)
PYTHONPATH=src python3 bin/reproduce-all

# GPU parity
modal run bin/gpu_energy.py

# GPU sum
TASK=sum modal run bin/gpu_energy.py
```

## Finding

**Sum is trivial for every method. Parity is hard for SGD.**

SGD solves sum in 1 epoch (0.8ms CPU) because the signal is first-order. Each secret bit contributes independently to the output. SGD solves parity in 40 epochs (142ms CPU) because it must discover the k-th order interaction through grokking.

On GPU, both tasks are slower than CPU. The tensors are too small for CUDA. But the ratio is revealing: parity-SGD is 10x slower on GPU, sum-SGD is 565x slower. Sum finishes in 1 epoch so the entire GPU time is kernel launch overhead for a single pass through 1000 samples.

GF(2) solves parity in 0.5ms but fails on sum entirely. This is correct: GF(2) exploits that parity is linear over the binary field. Sum is not parity over any field.

Sign SGD is unreliable on parity (3/5 seeds at best config) but solves sum 5/5 every time at 3.7ms mean.

**The 10ms budget**: On CPU, every sum method fits. On parity, only GF(2) (0.5ms) and KM (1.1ms) fit. SGD and Fourier don't fit for parity. Sign SGD is borderline.

## Files

- This document: `findings/exp_sum_vs_parity.md`
- CPU harness: `bin/reproduce-all`
- GPU script: `bin/gpu_energy.py`

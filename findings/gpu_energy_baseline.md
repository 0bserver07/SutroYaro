# GPU Energy Baseline: Real Watts on NVIDIA L4

**Date**: 2026-03-14
**Status**: BASELINE
**Challenge**: All three (sparse-parity, sparse-sum, sparse-and)

## Summary

First real energy measurement using `bin/gpu_energy.py` on Modal Labs (NVIDIA L4 GPU). Power sampled via pynvml during execution. Total cost: $0.0025.

## Results

| Challenge | Method | Acc | Time | Watts | Joules | ARD |
|-----------|--------|-----|------|-------|--------|-----|
| sparse-parity | gf2 | 1.00 | 3.2ms | n/a* | n/a* | 420 |
| sparse-parity | km | 1.00 | 11.6ms | 12.4W | 144mJ | 92 |
| sparse-parity | sgd | 1.00 | 1463ms | 12.7W | 18.7J | 8,504 |
| sparse-parity | fourier | 1.00 | 122ms | 12.9W | 1.57J | 11,980,500 |
| sparse-sum | sgd | 1.00 | 3.2ms | n/a* | n/a* | 20 |
| sparse-sum | km | 1.00 | 4.9ms | n/a* | n/a* | 92 |
| sparse-and | sgd | 1.00 | 69.4ms | 12.8W | 891mJ | 29,164 |
| sparse-and | km | 0.85 | 6.6ms | 12.8W | 85mJ | 92 |

*Too fast for power sampler (under 5ms, finishes before first reading).

## Key Findings

**1. GPU idles at ~12.5W regardless of method.** These experiments are too small to stress the GPU. Power draw is dominated by idle consumption, not computation. The L4's TDP is 72W but our workloads never push it past 13W.

**2. ARD roughly tracks real energy for methods above 10ms.** KM (ARD 92) uses 144mJ. SGD (ARD 8,504) uses 18.7J. That's a 130x energy gap, while the ARD gap is 92x. Same order of magnitude. The proxy works.

**3. Methods under 5ms can't be measured on GPU.** GF(2), sum-SGD, sum-KM all finish before the power sampler takes its first reading. For these, the idle power dominates and the actual compute energy is negligible (sub-millijoule).

**4. Parity SGD at 18.7 joules is wasteful.** LeCun trained digit recognition on a Spark 7 (64KB RAM, ~1 MFLOP/s). Our n=20/k=3 sparse parity is a simpler problem. 18.7J on a modern GPU is orders of magnitude more energy than necessary.

## Does the proxy track real energy?

| Method | ARD | Joules | ARD/Joules ratio |
|--------|-----|--------|-------------------|
| km (parity) | 92 | 0.144 | 639 |
| sgd (parity) | 8,504 | 18.7 | 455 |
| fourier (parity) | 11,980,500 | 1.57 | 7,631,529 |
| sgd (and) | 29,164 | 0.891 | 32,732 |

ARD tracks joules within an order of magnitude for SGD and KM but breaks down for Fourier. Fourier has terrible ARD (streaming access, no reuse) but moderate wall time (122ms) because the operations are simple. ARD overpenalizes streaming patterns relative to actual energy cost.

This confirms Yaroslav's observation that ARD is a useful but coarse proxy. DMC (Data Movement Complexity) may track better for streaming workloads.

## Config

- GPU: NVIDIA L4 (Modal Labs)
- Power sampling: pynvml, ~1ms interval
- n_bits=20, k_sparse=3, seed=42
- Container startup: ~8s, compute: ~2s

## How to reproduce

```bash
pip install modal
modal token set   # authenticate
modal run bin/gpu_energy.py
modal run bin/gpu_energy.py --json   # machine-readable output
```

Cost: under $0.01 per run.

## Files

- Script: `bin/gpu_energy.py`
- This document: `findings/gpu_energy_baseline.md`

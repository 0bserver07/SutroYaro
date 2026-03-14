#!/usr/bin/env python3
"""
Measure actual GPU energy (joules) for harness methods via Modal Labs.

Runs experiments on a cloud NVIDIA L4 GPU, samples power draw via pynvml
during execution, and reports joules per method.

Usage:
    modal run bin/gpu_energy.py
    modal run bin/gpu_energy.py --challenge sparse-sum
    modal run bin/gpu_energy.py --method gf2 --method km

Prerequisites:
    pip install modal
    modal token set  (authenticate with Modal Labs)

Cost: ~$0.003 per run (L4 at $0.84/hr, ~12s container time)
"""

import modal
import time
import json
import os

app = modal.App("sutro-gpu-energy")

# Copy src/ into the image at build time
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pynvml")
    .add_local_dir(src_path, "/root/src")
)

# Modal L4 pricing
L4_COST_PER_HOUR = 0.84
L4_COST_PER_SEC = L4_COST_PER_HOUR / 3600

# Default experiments
DEFAULT_EXPERIMENTS = [
    {"challenge": "sparse-parity", "method": "gf2"},
    {"challenge": "sparse-parity", "method": "km"},
    {"challenge": "sparse-parity", "method": "sgd"},
    {"challenge": "sparse-parity", "method": "fourier"},
    {"challenge": "sparse-sum", "method": "sgd"},
    {"challenge": "sparse-sum", "method": "km"},
    {"challenge": "sparse-and", "method": "sgd"},
    {"challenge": "sparse-and", "method": "km"},
]


@app.function(gpu="L4", image=image, timeout=120)
def run_with_energy(experiments, n_bits=20, k_sparse=3, seed=42, power_sample_ms=1):
    """Run experiments on GPU and measure actual power draw via pynvml."""
    import sys
    import numpy as np
    import threading

    sys.path.insert(0, "/root/src")
    from harness import measure_sparse_parity, measure_sparse_sum, measure_sparse_and

    dispatch = {
        "sparse-parity": measure_sparse_parity,
        "sparse-sum": measure_sparse_sum,
        "sparse-and": measure_sparse_and,
    }

    # Power monitoring
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        has_pynvml = True
        print(f"GPU: {gpu_name}")
    except Exception as e:
        print(f"pynvml not available: {e}")
        has_pynvml = False
        handle = None
        gpu_name = "unknown"

    results = []

    for exp in experiments:
        challenge = exp["challenge"]
        method = exp["method"]
        measure_fn = dispatch.get(challenge)

        if measure_fn is None:
            results.append({"challenge": challenge, "method": method,
                           "error": f"Unknown challenge: {challenge}"})
            continue

        # Power sampling thread
        power_readings = []
        stop_sampling = threading.Event()

        def sample_power():
            while not stop_sampling.is_set():
                try:
                    mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_readings.append((time.perf_counter(), mw / 1000.0))
                except Exception:
                    pass
                time.sleep(power_sample_ms / 1000.0)

        if has_pynvml:
            sampler = threading.Thread(target=sample_power, daemon=True)
            sampler.start()

        # Run experiment
        start = time.perf_counter()
        result = measure_fn(method=method, n_bits=n_bits, k_sparse=k_sparse, seed=seed)
        elapsed = time.perf_counter() - start

        if has_pynvml:
            stop_sampling.set()
            sampler.join(timeout=1)

        # Calculate energy
        if power_readings and len(power_readings) >= 2:
            watts = [w for _, w in power_readings]
            avg_watts = sum(watts) / len(watts)
            joules = avg_watts * elapsed
        else:
            avg_watts = None
            joules = None

        results.append({
            "challenge": challenge,
            "method": method,
            "accuracy": result.get("accuracy", 0),
            "time_s": round(elapsed, 6),
            "avg_watts": round(avg_watts, 2) if avg_watts else None,
            "joules": round(joules, 6) if joules else None,
            "ard": result.get("ard"),
            "dmc": result.get("dmc"),
            "total_floats": result.get("total_floats"),
            "power_samples": len(power_readings),
            "error": result.get("error"),
        })

        status = f"acc={result.get('accuracy', 0)}"
        if joules is not None:
            status += f" {joules*1e6:.1f}uJ {avg_watts:.1f}W"
        else:
            status += f" {elapsed*1000:.1f}ms"
        print(f"  {challenge:18s} {method:10s} {status}")

    if has_pynvml:
        pynvml.nvmlShutdown()

    return {"gpu": gpu_name, "results": results}


@app.local_entrypoint()
def main():
    import argparse

    parser = argparse.ArgumentParser(description="GPU energy measurement via Modal")
    parser.add_argument("--challenge", action="append",
                       help="Challenge(s) to run (can repeat)")
    parser.add_argument("--method", action="append",
                       help="Method(s) to run (can repeat)")
    parser.add_argument("--all", action="store_true",
                       help="Run all default experiments")
    parser.add_argument("--n_bits", type=int, default=20)
    parser.add_argument("--k_sparse", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true",
                       help="Output raw JSON")

    args, _ = parser.parse_known_args()

    if args.all or (not args.challenge and not args.method):
        experiments = DEFAULT_EXPERIMENTS
    else:
        experiments = []
        challenges = args.challenge or ["sparse-parity"]
        methods = args.method or ["gf2", "km", "sgd"]
        for c in challenges:
            for m in methods:
                experiments.append({"challenge": c, "method": m})

    print(f"Running {len(experiments)} experiments on Modal (L4 GPU)...")
    print()

    wall_start = time.time()

    result = run_with_energy.remote(
        experiments,
        n_bits=args.n_bits,
        k_sparse=args.k_sparse,
        seed=args.seed,
    )

    wall_elapsed = time.time() - wall_start
    gpu_cost = wall_elapsed * L4_COST_PER_SEC

    if args.json:
        result["wall_time"] = wall_elapsed
        result["estimated_cost"] = round(gpu_cost, 4)
        print(json.dumps(result, indent=2))
        return

    print()
    print(f"GPU: {result['gpu']}")
    print()
    print(f"{'Challenge':<18} {'Method':<10} {'Acc':>6} {'Time':>10} "
          f"{'Watts':>8} {'Joules':>12} {'ARD':>10}")
    print("-" * 80)

    for r in result["results"]:
        if r.get("error"):
            print(f"{r['challenge']:<18} {r['method']:<10} {'SKIP':>6} "
                  f"{'':>10} {'':>8} {'':>12} {'':>10}  {r['error']}")
            continue

        time_str = f"{r['time_s']*1000:.1f}ms"
        watts_str = f"{r['avg_watts']:.1f}W" if r.get('avg_watts') else "n/a"
        joules_str = f"{r['joules']*1e6:.1f}uJ" if r.get('joules') else "n/a"
        ard_str = f"{r['ard']:.0f}" if r.get('ard') else "n/a"

        print(f"{r['challenge']:<18} {r['method']:<10} {r['accuracy']:>6.2f} "
              f"{time_str:>10} {watts_str:>8} {joules_str:>12} {ard_str:>10}")

    print()
    print(f"Wall time: {wall_elapsed:.1f}s (includes container startup)")
    print(f"Estimated cost: ${gpu_cost:.4f} (L4 @ ${L4_COST_PER_HOUR}/hr)")

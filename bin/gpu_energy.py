#!/usr/bin/env python3
"""
Measure GPU compute time and cost for harness methods via Modal Labs.

Runs sparse parity methods on an NVIDIA L4 GPU using PyTorch CUDA,
matching Yaroslav's gpu_toy.py approach. Reports wall time, TFLOPS,
and estimated cost per method.

Usage:
    modal run bin/gpu_energy.py

Prerequisites:
    pip install modal
    modal token set

Cost: ~$0.003 per run (L4 at $0.84/hr, ~12s container time)
"""

import modal
import time
import json
import os

app = modal.App("sutro-gpu-energy")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
)

L4_COST_PER_HOUR = 0.84
L4_COST_PER_SEC = L4_COST_PER_HOUR / 3600


@app.function(gpu="L4", image=image, timeout=120)
def run_gpu(n_bits=20, k_sparse=3, n_train=1000, seed=42):
    """Run sparse parity methods on GPU via PyTorch CUDA."""
    import torch
    import numpy as np
    import subprocess

    device = torch.device("cuda")

    result = subprocess.run(["nvidia-smi", "--query-gpu=name,power.draw",
                            "--format=csv,noheader"], capture_output=True, text=True)
    print(f"GPU: {result.stdout.strip()}")
    print(f"PyTorch: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}")

    # Generate data
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())

    x_np = rng.choice([-1.0, 1.0], size=(n_train, n_bits)).astype(np.float32)
    y_np = np.prod(x_np[:, secret], axis=1).astype(np.float32)
    x_te_np = rng.choice([-1.0, 1.0], size=(200, n_bits)).astype(np.float32)
    y_te_np = np.prod(x_te_np[:, secret], axis=1).astype(np.float32)

    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    x_te = torch.from_numpy(x_te_np).to(device)
    y_te = torch.from_numpy(y_te_np).to(device)

    results = []

    # --- GF(2) on GPU ---
    # GF(2) is integer XOR, not matrix multiply. Runs on CPU because
    # CUDA doesn't help with sequential row reduction.
    torch.cuda.synchronize()
    start = time.perf_counter()

    A = ((x_np[:n_bits+1] + 1) / 2).astype(np.uint8)
    b = ((y_np[:n_bits+1] + 1) / 2).astype(np.uint8)
    found = None

    for b_try in [b, (1 - b).astype(np.uint8)]:
        aug = np.hstack([A.copy(), b_try.reshape(-1, 1)])
        row = 0
        for col in range(n_bits):
            pivot = None
            for r in range(row, len(aug)):
                if aug[r, col] == 1:
                    pivot = r
                    break
            if pivot is None:
                continue
            aug[[row, pivot]] = aug[[pivot, row]]
            for r in range(len(aug)):
                if r != row and aug[r, col] == 1:
                    aug[r] = aug[r] ^ aug[row]
            row += 1

        sol = np.zeros(n_bits, dtype=np.uint8)
        for i in range(min(row, n_bits)):
            sol[i] = aug[i, -1]
        candidate = sorted(np.where(sol == 1)[0].tolist())
        if candidate:
            y_check = np.prod(x_np[:n_bits+1][:, candidate], axis=1)
            if np.allclose(y_check, y_np[:n_bits+1]):
                found = candidate
                break

    if found:
        y_pred = np.prod(x_te_np[:, found], axis=1)
        gf2_acc = float(np.mean(y_pred == y_te_np))
    else:
        gf2_acc = 0.0

    elapsed = time.perf_counter() - start
    results.append({"method": "gf2", "accuracy": round(gf2_acc, 4),
                    "time_s": round(elapsed, 6), "note": "CPU (sequential XOR)"})
    print(f"  gf2: acc={gf2_acc:.2f} {elapsed*1000:.1f}ms (CPU)")

    # --- SGD on GPU (PyTorch, CUDA matmuls) ---
    # Match the numpy harness: hinge loss, same init scale, same hyperparams
    hidden = 200
    torch.manual_seed(seed + 1)
    W1 = torch.randn(hidden, n_bits, device=device) * np.sqrt(2.0 / n_bits)
    b1 = torch.zeros(hidden, device=device)
    W2 = torch.randn(1, hidden, device=device) * np.sqrt(2.0 / hidden)
    b2 = torch.zeros(1, device=device)
    lr = 0.1
    wd = 0.01
    batch_size = 32

    torch.cuda.synchronize()
    start = time.perf_counter()

    best_acc = 0.0
    final_epoch = 0
    for epoch in range(200):
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, batch_size):
            idx = perm[s:s+batch_size]
            xb = x[idx]
            yb = y[idx].unsqueeze(1)

            # Forward
            h_pre = xb @ W1.t() + b1
            h = torch.relu(h_pre)
            out = h @ W2.t() + b2

            # Hinge loss gradient: max(0, 1 - y*f(x))
            margin = yb * out
            mask = (margin < 1.0).float()
            d_out = -yb * mask / len(idx)

            # Backward
            dW2 = d_out.t() @ h
            db2 = d_out.sum(dim=0)
            d_h = d_out @ W2
            d_h = d_h * (h_pre > 0).float()
            dW1 = d_h.t() @ xb
            db1 = d_h.sum(dim=0)

            # Update with weight decay
            W1 = W1 - lr * (dW1 + wd * W1)
            b1 = b1 - lr * db1
            W2 = W2 - lr * (dW2 + wd * W2)
            b2 = b2 - lr * db2

        with torch.no_grad():
            h_te = torch.relu(x_te @ W1.t() + b1)
            out_te = (h_te @ W2.t() + b2).squeeze()
            pred = torch.sign(out_te)
            acc = float((pred == y_te).float().mean())
            best_acc = max(best_acc, acc)
            final_epoch = epoch + 1
            if acc >= 1.0:
                break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    results.append({"method": "sgd", "accuracy": round(best_acc, 4),
                    "time_s": round(elapsed, 6), "epochs": final_epoch,
                    "note": "GPU (PyTorch CUDA, hinge loss)"})
    print(f"  sgd: acc={best_acc:.2f} {elapsed*1000:.1f}ms {final_epoch} epochs (GPU)")

    # --- KM influence on GPU ---
    torch.cuda.synchronize()
    start = time.perf_counter()

    n_inf = 20
    influences = torch.zeros(n_bits, device=device)
    rng_inf = torch.Generator(device=device).manual_seed(seed + 500)

    for i in range(n_bits):
        xb = torch.where(
            torch.rand(n_inf, n_bits, device=device, generator=rng_inf) > 0.5,
            torch.ones(1, device=device),
            -torch.ones(1, device=device)
        )
        secret_idx = torch.tensor(secret, device=device)
        y_orig = torch.prod(xb[:, secret_idx], dim=1)
        xb_flip = xb.clone()
        xb_flip[:, i] *= -1
        y_flip = torch.prod(xb_flip[:, secret_idx], dim=1)
        influences[i] = (y_orig != y_flip).float().mean()

    top_k = torch.argsort(influences)[-k_sparse:].sort().values.tolist()
    top_k_idx = torch.tensor(top_k, device=device)
    y_pred = torch.prod(x_te[:, top_k_idx], dim=1)
    km_acc = float((y_pred == y_te).float().mean())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    results.append({"method": "km", "accuracy": round(km_acc, 4),
                    "time_s": round(elapsed, 6), "note": "GPU (PyTorch)"})
    print(f"  km: acc={km_acc:.2f} {elapsed*1000:.1f}ms (GPU)")

    return {
        "gpu": torch.cuda.get_device_name(0),
        "secret": secret,
        "results": results,
    }


@app.local_entrypoint()
def main():
    print("Running sparse parity on Modal (L4 GPU, PyTorch CUDA)...")
    print()

    wall_start = time.time()
    result = run_gpu.remote()
    wall_elapsed = time.time() - wall_start
    gpu_cost = wall_elapsed * L4_COST_PER_SEC

    print()
    print(f"GPU: {result['gpu']}")
    print(f"Secret: {result['secret']}")
    print()
    print(f"{'Method':<10} {'Acc':>6} {'Time':>10} {'Note'}")
    print("-" * 55)

    for r in result["results"]:
        time_str = f"{r['time_s']*1000:.1f}ms"
        extra = f" ({r['epochs']} epochs)" if r.get('epochs') else ""
        print(f"{r['method']:<10} {r['accuracy']:>6.2f} {time_str:>10} {r.get('note','')}{extra}")

    print()
    print(f"Wall time: {wall_elapsed:.1f}s (includes container startup)")
    print(f"Estimated cost: ${gpu_cost:.4f} (L4 @ ${L4_COST_PER_HOUR}/hr)")

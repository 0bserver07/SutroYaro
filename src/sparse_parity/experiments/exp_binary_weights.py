#!/usr/bin/env python3
"""
Experiment: Binary Weights (BinaryConnect) for Sparse Parity

Approach #13: Mixed Precision / Binary Weights (Courbariaux et al. 2015).
Train with binary weights (+1/-1). Sparse parity with sign-encoded inputs
is fundamentally a binary operation. A network with binary weights and sign
activation can represent parity exactly. Binary ops use ~30x less energy
than float32. Memory footprint shrinks 32x.

Method: maintain float32 "shadow" weights for gradient accumulation, but
binarize (sign function) for forward/backward. Use straight-through
estimator for gradients through the sign function.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_binary_weights.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sparse_parity.tracker import MemTracker


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n_bits, k_sparse, n_train, n_test, seed=42):
    """Generate sparse parity train/test data. Same secret for both.
    Returns x_tr, y_tr, x_te, y_te, secret."""
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())

    def make(n):
        x = rng.choice([-1.0, 1.0], size=(n, n_bits))
        y = np.prod(x[:, secret], axis=1)
        return x, y

    x_tr, y_tr = make(n_train)
    x_te, y_te = make(n_test)
    return x_tr, y_tr, x_te, y_te, secret


# =============================================================================
# BINARY WEIGHTS TRAINING (BinaryConnect with straight-through estimator)
# =============================================================================

def train_binary(x_tr, y_tr, x_te, y_te, n_bits, hidden, lr, wd,
                 max_epochs, seed=42, verbose=True, batch_size=32,
                 track_ard=False):
    """
    BinaryConnect training:
    - Shadow weights W1_float, W2_float in float32 (for gradient accumulation)
    - Binary weights W1_bin = sign(W1_float), W2_bin = sign(W2_float)
    - Forward: h = sign(W1_bin @ x + b1), y_hat = W2_bin @ h + b2
    - Backward: straight-through estimator (gradient passes through sign unchanged)
    - Update: W1_float -= lr * grad, then re-binarize
    """
    rng = np.random.RandomState(seed + 1)

    # Initialize shadow weights (float32)
    std1 = np.sqrt(2.0 / n_bits)
    std2 = np.sqrt(2.0 / hidden)
    W1_float = rng.randn(hidden, n_bits).astype(np.float32) * std1
    b1 = np.zeros(hidden, dtype=np.float32)
    W2_float = rng.randn(1, hidden).astype(np.float32) * std2
    b2 = np.zeros(1, dtype=np.float32)

    # Set up tracker for one step if requested
    tracker = None
    tracker_result = None
    tracked = False

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1
    n_train = x_tr.shape[0]

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(n_train)
        rng.shuffle(idx)

        for b_start in range(0, n_train, batch_size):
            b_end = min(b_start + batch_size, n_train)
            xb = x_tr[idx[b_start:b_end]]
            yb = y_tr[idx[b_start:b_end]]
            bs = xb.shape[0]

            # Set up tracker for the first mini-batch only
            if track_ard and not tracked:
                tracker = MemTracker()
                # Initial writes of params and data into memory
                tracker.write('W1_float', hidden * n_bits)
                tracker.write('W2_float', hidden)
                tracker.write('b1', hidden)
                tracker.write('b2', 1)
                tracker.write('x_batch', bs * n_bits)
                tracker.write('y_batch', bs)

            # === BINARIZE (sign of shadow weights) ===
            W1_bin = np.sign(W1_float)
            W2_bin = np.sign(W2_float)
            # Replace zeros with +1
            W1_bin[W1_bin == 0] = 1.0
            W2_bin[W2_bin == 0] = 1.0

            if tracker:
                # Reading shadow weights to binarize
                tracker.read('W1_float', hidden * n_bits)
                tracker.read('W2_float', hidden)
                # Writing binary weights — counted as 1/32 of float size
                tracker.write('W1_bin', (hidden * n_bits) // 32)
                tracker.write('W2_bin', hidden // 32)

            # === FORWARD ===
            if tracker:
                tracker.read('W1_bin', (hidden * n_bits) // 32)
                tracker.read('x_batch', bs * n_bits)
                tracker.read('b1', hidden)

            h_pre = xb @ W1_bin.T + b1  # (bs, hidden)

            if tracker:
                tracker.write('h_pre', bs * hidden)
                tracker.read('h_pre', bs * hidden)

            # Sign activation instead of ReLU
            h = np.sign(h_pre)
            h[h == 0] = 1.0

            if tracker:
                tracker.write('h', bs * hidden)
                tracker.read('h', bs * hidden)
                tracker.read('W2_bin', hidden // 32)
                tracker.read('b2', 1)

            out = (h @ W2_bin.T + b2).ravel()  # (bs,)

            if tracker:
                tracker.write('out', bs)

            # === HINGE LOSS + MASK ===
            margin = out * yb
            mask = margin < 1.0
            if not np.any(mask):
                if tracker:
                    tracked = True
                    tracker_result = tracker.to_json()
                    tracker = None
                continue

            xm = xb[mask]
            ym = yb[mask]
            hm = h[mask]
            h_pre_m = h_pre[mask]
            ms = xm.shape[0]

            if tracker:
                tracker.read('out', bs)
                tracker.read('y_batch', bs)

            # === BACKWARD with straight-through estimator ===
            # dL/dout = -y for violated samples
            dout = -ym  # (ms,)

            if tracker:
                tracker.write('dout', ms)

            # dW2 = dout^T @ h  (gradient w.r.t. W2_bin, but applied to W2_float)
            # Straight-through: treat sign as identity for gradient
            dW2 = dout[:, None] * hm  # (ms, hidden)
            db2 = dout.sum()

            if tracker:
                tracker.read('dout', ms)
                tracker.read('h', bs * hidden)
                tracker.write('dW2', hidden)
                tracker.write('db2', 1)

            # dh = W2_bin^T * dout  -> (ms, hidden)
            dh = dout[:, None] * W2_bin  # (ms, hidden)

            if tracker:
                tracker.read('W2_bin', hidden // 32)
                tracker.read('dout', ms)
                tracker.write('dh', ms * hidden)

            # Straight-through estimator for sign activation:
            # gradient passes through sign unchanged (STE)
            dh_pre = dh  # no gating, straight-through

            if tracker:
                tracker.read('dh', ms * hidden)
                tracker.write('dh_pre', ms * hidden)

            # dW1 = dh_pre^T @ x  -> (hidden, n_bits)
            dW1 = dh_pre.T @ xm  # (hidden, n_bits)
            db1_grad = dh_pre.sum(axis=0)  # (hidden,)

            if tracker:
                tracker.read('dh_pre', ms * hidden)
                tracker.read('x_batch', bs * n_bits)
                tracker.write('dW1', hidden * n_bits)
                tracker.write('db1_grad', hidden)

            # === UPDATE shadow weights ===
            if tracker:
                tracker.read('dW1', hidden * n_bits)
                tracker.read('W1_float', hidden * n_bits)

            W1_float -= lr * (dW1 / bs + wd * W1_float)

            if tracker:
                tracker.write('W1_float', hidden * n_bits)
                tracker.read('db1_grad', hidden)
                tracker.read('b1', hidden)

            b1 -= lr * (db1_grad / bs + wd * b1)

            if tracker:
                tracker.write('b1', hidden)
                tracker.read('dW2', hidden)
                tracker.read('W2_float', hidden)

            W2_float -= lr * (dW2.sum(axis=0, keepdims=True) / bs + wd * W2_float)

            if tracker:
                tracker.write('W2_float', hidden)
                tracker.read('db2', 1)
                tracker.read('b2', 1)

            b2 -= lr * (db2 / bs + wd * b2)

            if tracker:
                tracker.write('b2', 1)
                tracked = True
                tracker_result = tracker.to_json()
                tracker = None

        # === EVALUATE (use binary weights) ===
        W1_eval = np.sign(W1_float)
        W2_eval = np.sign(W2_float)
        W1_eval[W1_eval == 0] = 1.0
        W2_eval[W2_eval == 0] = 1.0

        te_h = np.sign(x_te @ W1_eval.T + b1)
        te_h[te_h == 0] = 1.0
        te_out = (te_h @ W2_eval.T + b2).ravel()
        te_acc = np.mean(np.sign(te_out) == y_te)

        tr_h = np.sign(x_tr @ W1_eval.T + b1)
        tr_h[tr_h == 0] = 1.0
        tr_out = (tr_h @ W2_eval.T + b2).ravel()
        tr_acc = np.mean(np.sign(tr_out) == y_tr)

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch

        if verbose and (epoch % 20 == 0 or epoch == 1 or te_acc >= 1.0):
            print(f"    epoch {epoch:>4}: train={tr_acc:.0%} test={te_acc:.0%}")

        if te_acc >= 1.0:
            break

    elapsed = time.time() - start
    if verbose:
        status = "SOLVED" if best_acc >= 0.99 else f"{best_acc:.1%}"
        print(f"  Result: {status} in {elapsed:.2f}s ({epoch} epochs)")

    return {
        'method': 'binary_weights',
        'best_test_acc': round(float(best_acc), 4),
        'solve_epoch': solve_epoch,
        'total_epochs': epoch,
        'elapsed_s': round(elapsed, 3),
        'n_bits': n_bits,
        'hidden': hidden,
        'lr': lr,
        'wd': wd,
        'tracker': tracker_result,
    }


# =============================================================================
# FLOAT32 BASELINE (standard SGD with ReLU for comparison)
# =============================================================================

def train_float32(x_tr, y_tr, x_te, y_te, n_bits, hidden, lr, wd,
                  max_epochs, seed=42, verbose=True, batch_size=32,
                  track_ard=False):
    """Standard float32 training with ReLU activation for baseline comparison."""
    rng = np.random.RandomState(seed + 1)

    std1 = np.sqrt(2.0 / n_bits)
    std2 = np.sqrt(2.0 / hidden)
    W1 = rng.randn(hidden, n_bits).astype(np.float32) * std1
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.randn(1, hidden).astype(np.float32) * std2
    b2 = np.zeros(1, dtype=np.float32)

    tracker = None
    tracker_result = None
    tracked = False

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1
    n_train = x_tr.shape[0]

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(n_train)
        rng.shuffle(idx)

        for b_start in range(0, n_train, batch_size):
            b_end = min(b_start + batch_size, n_train)
            xb = x_tr[idx[b_start:b_end]]
            yb = y_tr[idx[b_start:b_end]]
            bs = xb.shape[0]

            if track_ard and not tracked:
                tracker = MemTracker()
                tracker.write('W1', hidden * n_bits)
                tracker.write('W2', hidden)
                tracker.write('b1', hidden)
                tracker.write('b2', 1)
                tracker.write('x_batch', bs * n_bits)
                tracker.write('y_batch', bs)

            # Forward
            if tracker:
                tracker.read('W1', hidden * n_bits)
                tracker.read('x_batch', bs * n_bits)
                tracker.read('b1', hidden)

            h_pre = xb @ W1.T + b1
            h = np.maximum(h_pre, 0)

            if tracker:
                tracker.write('h_pre', bs * hidden)
                tracker.write('h', bs * hidden)
                tracker.read('h', bs * hidden)
                tracker.read('W2', hidden)
                tracker.read('b2', 1)

            out = (h @ W2.T + b2).ravel()

            if tracker:
                tracker.write('out', bs)

            margin = out * yb
            mask = margin < 1.0
            if not np.any(mask):
                if tracker:
                    tracked = True
                    tracker_result = tracker.to_json()
                    tracker = None
                continue

            xm = xb[mask]
            ym = yb[mask]
            hm = h[mask]
            h_pre_m = h_pre[mask]
            ms = xm.shape[0]

            if tracker:
                tracker.read('out', bs)
                tracker.read('y_batch', bs)

            dout = -ym
            dW2 = dout[:, None] * hm
            db2 = dout.sum()
            dh = dout[:, None] * W2
            dh_pre = dh * (h_pre_m > 0)
            dW1 = dh_pre.T @ xm
            db1_grad = dh_pre.sum(axis=0)

            if tracker:
                tracker.write('dout', ms)
                tracker.read('dout', ms)
                tracker.read('h', bs * hidden)
                tracker.write('dW2', hidden)
                tracker.write('db2', 1)
                tracker.read('W2', hidden)
                tracker.write('dh', ms * hidden)
                tracker.read('dh', ms * hidden)
                tracker.read('h_pre', bs * hidden)
                tracker.write('dh_pre', ms * hidden)
                tracker.read('dh_pre', ms * hidden)
                tracker.read('x_batch', bs * n_bits)
                tracker.write('dW1', hidden * n_bits)
                tracker.write('db1_grad', hidden)

            # Update
            if tracker:
                tracker.read('dW1', hidden * n_bits)
                tracker.read('W1', hidden * n_bits)

            W1 -= lr * (dW1 / bs + wd * W1)

            if tracker:
                tracker.write('W1', hidden * n_bits)
                tracker.read('db1_grad', hidden)
                tracker.read('b1', hidden)

            b1 -= lr * (db1_grad / bs + wd * b1)

            if tracker:
                tracker.write('b1', hidden)
                tracker.read('dW2', hidden)
                tracker.read('W2', hidden)

            W2 -= lr * (dW2.sum(axis=0, keepdims=True) / bs + wd * W2)

            if tracker:
                tracker.write('W2', hidden)
                tracker.read('db2', 1)
                tracker.read('b2', 1)

            b2 -= lr * (db2 / bs + wd * b2)

            if tracker:
                tracker.write('b2', 1)
                tracked = True
                tracker_result = tracker.to_json()
                tracker = None

        # Evaluate
        te_out = (np.maximum(x_te @ W1.T + b1, 0) @ W2.T + b2).ravel()
        te_acc = np.mean(np.sign(te_out) == y_te)
        tr_out = (np.maximum(x_tr @ W1.T + b1, 0) @ W2.T + b2).ravel()
        tr_acc = np.mean(np.sign(tr_out) == y_tr)

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch

        if verbose and (epoch % 20 == 0 or epoch == 1 or te_acc >= 1.0):
            print(f"    epoch {epoch:>4}: train={tr_acc:.0%} test={te_acc:.0%}")

        if te_acc >= 1.0:
            break

    elapsed = time.time() - start
    if verbose:
        status = "SOLVED" if best_acc >= 0.99 else f"{best_acc:.1%}"
        print(f"  Result: {status} in {elapsed:.2f}s ({epoch} epochs)")

    return {
        'method': 'float32_baseline',
        'best_test_acc': round(float(best_acc), 4),
        'solve_epoch': solve_epoch,
        'total_epochs': epoch,
        'elapsed_s': round(elapsed, 3),
        'n_bits': n_bits,
        'hidden': hidden,
        'lr': lr,
        'wd': wd,
        'tracker': tracker_result,
    }


# =============================================================================
# CONFIG RUNNER
# =============================================================================

def run_config(label, n_bits, k_sparse, hidden, lr, wd, max_epochs,
               n_train, n_test, seeds, batch_size=32, verbose=True):
    """Run binary weights and float32 baseline across seeds. Returns dict."""
    print(f"\n  Config: n={n_bits}, k={k_sparse}, hidden={hidden}, "
          f"lr={lr}, n_train={n_train}")

    binary_results = []
    float32_results = []

    for seed in seeds:
        x_tr, y_tr, x_te, y_te, secret = generate_data(
            n_bits, k_sparse, n_train, n_test, seed=seed
        )

        # Binary weights
        if verbose and seed == seeds[0]:
            print(f"\n  --- Binary Weights (seed={seed}) ---")
        r_bin = train_binary(
            x_tr, y_tr, x_te, y_te, n_bits, hidden, lr, wd,
            max_epochs, seed=seed, verbose=(verbose and seed == seeds[0]),
            batch_size=batch_size, track_ard=(seed == seeds[0])
        )
        r_bin['seed'] = seed
        r_bin['secret'] = secret
        binary_results.append(r_bin)

        if not verbose or seed != seeds[0]:
            status = "SOLVED" if r_bin['best_test_acc'] >= 0.99 else f"{r_bin['best_test_acc']:.1%}"
            print(f"    [binary] seed={seed}: {status} ep={r_bin['total_epochs']} ({r_bin['elapsed_s']:.2f}s)")

        # Float32 baseline
        if verbose and seed == seeds[0]:
            print(f"\n  --- Float32 Baseline (seed={seed}) ---")
        r_f32 = train_float32(
            x_tr, y_tr, x_te, y_te, n_bits, hidden, lr, wd,
            max_epochs, seed=seed, verbose=(verbose and seed == seeds[0]),
            batch_size=batch_size, track_ard=(seed == seeds[0])
        )
        r_f32['seed'] = seed
        r_f32['secret'] = secret
        float32_results.append(r_f32)

        if not verbose or seed != seeds[0]:
            status = "SOLVED" if r_f32['best_test_acc'] >= 0.99 else f"{r_f32['best_test_acc']:.1%}"
            print(f"    [float32] seed={seed}: {status} ep={r_f32['total_epochs']} ({r_f32['elapsed_s']:.2f}s)")

    return {
        'label': label,
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'hidden': hidden,
        'lr': lr,
        'wd': wd,
        'n_train': n_train,
        'n_test': n_test,
        'binary': binary_results,
        'float32': float32_results,
    }


# =============================================================================
# ENERGY PROXY COMPUTATION
# =============================================================================

def compute_energy_proxy(tracker_json, is_binary=False):
    """
    Compute energy proxy from tracker data.
    For binary weights, count binary reads as 1/32 of float reads.
    Returns dict with energy estimate.
    """
    if tracker_json is None:
        return {'energy_proxy': None}

    total = tracker_json.get('total_floats_accessed', 0)
    weighted_ard = tracker_json.get('weighted_ard', 0)

    # For binary: the binary weight reads are already counted as 1/32 size
    # in the tracker, so the energy proxy naturally reflects the savings
    energy_proxy = total * weighted_ard  # simple product as proxy

    return {
        'total_floats_accessed': total,
        'weighted_ard': weighted_ard,
        'energy_proxy': energy_proxy,
        'is_binary': is_binary,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: Binary Weights (BinaryConnect) for Sparse Parity")
    print("  Approach #13: Binary weights (+1/-1) with straight-through estimator")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # -------------------------------------------------------------------
    # Config 1: n=3, k=3 (sanity check — all bits are parity bits)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=3, k=3 (sanity check)")
    print("=" * 70)

    all_results['n3_k3'] = run_config(
        label='n3_k3_sanity',
        n_bits=3, k_sparse=3, hidden=100, lr=0.1, wd=0.01,
        max_epochs=200, n_train=100, n_test=100, seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 2: n=20, k=3 — hidden=200, lr=0.1 (known-good baseline config)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=20, k=3, hidden=200, lr=0.1 (proven baseline)")
    print("=" * 70)

    all_results['n20_k3_h200'] = run_config(
        label='n20_k3_h200',
        n_bits=20, k_sparse=3, hidden=200, lr=0.1, wd=0.01,
        max_epochs=200, n_train=1000, n_test=500, seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 3: n=20, k=3 — hidden=1000, lr=0.01
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 3: n=20, k=3, hidden=1000, lr=0.01")
    print("=" * 70)

    all_results['n20_k3_h1000_lr001'] = run_config(
        label='n20_k3_h1000_lr001',
        n_bits=20, k_sparse=3, hidden=1000, lr=0.01, wd=0.01,
        max_epochs=200, n_train=1000, n_test=500, seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 4: n=20, k=3 — hidden=1000, lr=0.1
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 4: n=20, k=3, hidden=1000, lr=0.1")
    print("=" * 70)

    all_results['n20_k3_h1000_lr01'] = run_config(
        label='n20_k3_h1000_lr01',
        n_bits=20, k_sparse=3, hidden=1000, lr=0.1, wd=0.01,
        max_epochs=200, n_train=1000, n_test=500, seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)

    header = (f"  {'Config':<25} | {'Method':<10} | {'Avg Acc':>8} | "
              f"{'Solved':>6} | {'Avg Epochs':>10} | {'Avg Time':>9} | {'ARD':>10}")
    print(header)
    print("  " + "-" * 88)

    for key, res in all_results.items():
        for method_key, method_label in [('binary', 'binary'), ('float32', 'float32')]:
            runs = res[method_key]
            avg_acc = np.mean([r['best_test_acc'] for r in runs])
            n_solved = sum(1 for r in runs if r['best_test_acc'] >= 0.99)
            avg_epochs = np.mean([r['total_epochs'] for r in runs])
            avg_time = np.mean([r['elapsed_s'] for r in runs])

            # Get ARD from first seed's tracker
            ard_str = "---"
            if runs[0].get('tracker'):
                ard = runs[0]['tracker'].get('weighted_ard', 0)
                ard_str = f"{ard:,.0f}"

            print(f"  {key:<25} | {method_label:<10} | {avg_acc:>8.1%} | "
                  f"{n_solved}/{len(runs):>4} | {avg_epochs:>10.0f} | "
                  f"{avg_time:>8.2f}s | {ard_str:>10}")

    # -------------------------------------------------------------------
    # Energy comparison
    # -------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  ENERGY PROXY COMPARISON")
    print("=" * 90)
    print(f"  {'Config':<25} | {'Binary Energy':>15} | {'Float32 Energy':>15} | {'Ratio':>8}")
    print("  " + "-" * 70)

    for key, res in all_results.items():
        bin_energy = compute_energy_proxy(res['binary'][0].get('tracker'), is_binary=True)
        f32_energy = compute_energy_proxy(res['float32'][0].get('tracker'), is_binary=False)

        if bin_energy['energy_proxy'] is not None and f32_energy['energy_proxy'] is not None:
            ratio = f32_energy['energy_proxy'] / bin_energy['energy_proxy'] if bin_energy['energy_proxy'] > 0 else float('inf')
            print(f"  {key:<25} | {bin_energy['energy_proxy']:>15,.0f} | "
                  f"{f32_energy['energy_proxy']:>15,.0f} | {ratio:>7.1f}x")
        else:
            print(f"  {key:<25} | {'---':>15} | {'---':>15} | {'---':>8}")

    # -------------------------------------------------------------------
    # Memory footprint comparison
    # -------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  MEMORY FOOTPRINT")
    print("=" * 90)
    for key, res in all_results.items():
        n = res['n_bits']
        h = res['hidden']
        float_params = h * n + h + h + 1  # W1 + b1 + W2 + b2
        float_bytes = float_params * 4  # float32 = 4 bytes
        # Binary: shadow weights (float32) + binary weights (1 bit each)
        binary_shadow_bytes = float_params * 4
        binary_weight_bytes = (h * n + h) // 8  # W1 + W2 in bits -> bytes
        # At inference, only binary weights needed
        inference_bytes = binary_weight_bytes + (h + 1) * 4  # binary W + float biases

        print(f"  {key}:")
        print(f"    Float32 inference: {float_bytes:,} bytes ({float_bytes/1024:.1f} KB)")
        print(f"    Binary inference:  {inference_bytes:,} bytes ({inference_bytes/1024:.1f} KB) "
              f"({float_bytes/inference_bytes:.1f}x smaller)")
        print(f"    Binary training:   {binary_shadow_bytes + binary_weight_bytes:,} bytes "
              f"(shadow + binary)")

    print("\n" + "=" * 70)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_binary_weights'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Make results JSON-serializable
    serializable = {}
    for key, res in all_results.items():
        serializable[key] = {
            k: v for k, v in res.items()
        }

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_binary_weights',
            'description': 'BinaryConnect (Courbariaux 2015): binary weights with straight-through estimator',
            'hypothesis': 'Binary weights can solve sparse parity with 32x less memory at inference',
            'approach': 'Binary weights (+1/-1) with float32 shadow weights for gradient accumulation',
            'configs': serializable,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

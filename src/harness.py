#!/usr/bin/env python3
"""
Locked evaluation harness for SutroYaro experiments.

DO NOT MODIFY THIS FILE IN EXPERIMENT PRs.
This is LAB.md rule #9. If you think the harness is wrong,
log a note in your findings doc. Do not fix it yourself.

Usage:
    PYTHONPATH=src python3 src/harness.py --method sgd --n_bits 20 --k_sparse 3
    PYTHONPATH=src python3 src/harness.py --method gf2 --n_bits 100 --k_sparse 10
"""

import sys
import os
import time
import json
import argparse

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from sparse_parity.config import Config
from sparse_parity.tracker import MemTracker


def measure_sparse_parity(method, n_bits=20, k_sparse=3, hidden=200,
                           lr=0.1, wd=0.01, batch_size=32, n_train=1000,
                           max_epochs=200, seed=42, **kwargs):
    """
    Run a sparse parity experiment and return standardized metrics.

    Returns dict with: accuracy, ard, dmc, time_s, total_floats, method, config
    """
    config = Config(
        n_bits=n_bits, k_sparse=k_sparse, hidden=hidden,
        lr=lr, wd=wd, batch_size=batch_size,
        n_train=n_train, n_test=200, max_epochs=max_epochs, seed=seed
    )

    start = time.perf_counter()

    if method == "sgd":
        result = _run_sgd(config, **kwargs)
    elif method == "gf2":
        result = _run_gf2(config, **kwargs)
    elif method == "km":
        result = _run_km(config, **kwargs)
    elif method == "fourier":
        result = _run_fourier(config, **kwargs)
    elif method == "smt":
        result = _run_smt(config, **kwargs)
    else:
        return {"error": f"Unknown method: {method}", "method": method}

    elapsed = time.perf_counter() - start

    result["time_s"] = round(elapsed, 6)
    result["method"] = method
    result["config"] = {
        "n_bits": n_bits, "k_sparse": k_sparse, "hidden": hidden,
        "lr": lr, "wd": wd, "batch_size": batch_size,
        "n_train": n_train, "max_epochs": max_epochs, "seed": seed,
    }

    return result


def _run_sgd(config, track_step=None, **_kwargs):
    """Standard SGD training with optional ARD tracking. Uses fast.py."""
    from sparse_parity.fast import train, generate

    r = train(config, verbose=False)

    result = {
        "accuracy": round(r["best_test_acc"], 4),
        "epochs": r["total_epochs"],
    }

    # Track one step for ARD measurement if training succeeded
    if track_step is not None or r["best_test_acc"] >= 0.95:
        import numpy as np
        x_tr, y_tr, x_te, y_te, secret = generate(config)

        rng = np.random.RandomState(config.seed + 1)
        std1 = np.sqrt(2.0 / config.n_bits)
        std2 = np.sqrt(2.0 / config.hidden)
        W1 = rng.randn(config.hidden, config.n_bits) * std1
        b1 = np.zeros(config.hidden)
        W2 = rng.randn(1, config.hidden) * std2
        b2 = np.zeros(1)

        tracker = MemTracker()
        _tracked_sgd_step(x_tr, y_tr, W1, b1, W2, b2, config, tracker)
        s = tracker.summary()
        result["ard"] = round(s["weighted_ard"], 1)
        result["dmc"] = round(s["dmc"], 1)
        result["total_floats"] = s["total_floats_accessed"]

    return result


def _tracked_sgd_step(x_tr, y_tr, W1, b1, W2, b2, config, tracker):
    """One tracked training step for ARD measurement."""
    import numpy as np

    n = min(1, len(x_tr))  # single sample for tracking
    x = x_tr[0:1]
    y = y_tr[0:1]

    # Forward
    tracker.write("x", x.size)
    tracker.write("W1", W1.size)
    tracker.write("b1", b1.size)
    tracker.read("x")
    tracker.read("W1")
    tracker.read("b1")
    h_pre = x @ W1.T + b1
    tracker.write("h_pre", h_pre.size)

    tracker.read("h_pre")
    h = np.maximum(0, h_pre)
    tracker.write("h", h.size)

    tracker.write("W2", W2.size)
    tracker.write("b2", b2.size)
    tracker.read("h")
    tracker.read("W2")
    tracker.read("b2")
    out = h @ W2.T + b2
    tracker.write("out", out.size)

    # Backward
    tracker.read("out")
    y_float = y.reshape(-1, 1).astype(np.float32)
    d_out = out - y_float
    tracker.write("d_out", d_out.size)

    tracker.read("d_out")
    tracker.read("W2")
    tracker.read("h")
    dW2 = d_out.T @ h
    db2 = d_out.sum(axis=0)
    tracker.write("dW2", dW2.size)
    tracker.write("db2", db2.size)

    tracker.read("d_out")
    tracker.read("W2")
    d_h = d_out @ W2
    tracker.write("d_h", d_h.size)

    tracker.read("d_h")
    tracker.read("h_pre")
    d_h_pre = d_h * (h_pre > 0).astype(np.float32)
    tracker.write("d_h_pre", d_h_pre.size)

    tracker.read("d_h_pre")
    tracker.read("x")
    dW1 = d_h_pre.T @ x
    db1 = d_h_pre.sum(axis=0)
    tracker.write("dW1", dW1.size)
    tracker.write("db1", db1.size)

    # Update
    tracker.read("W1")
    tracker.read("dW1")
    tracker.read("W2")
    tracker.read("dW2")
    tracker.read("b1")
    tracker.read("db1")
    tracker.read("b2")
    tracker.read("db2")


def _run_gf2(config, **kwargs):
    """GF(2) Gaussian elimination. Tries both b and 1-b for even/odd k."""
    import numpy as np

    rng = np.random.RandomState(config.seed)
    secret = sorted(rng.choice(config.n_bits, config.k_sparse, replace=False).tolist())

    n_samples = config.n_bits + 1
    x = rng.choice([-1.0, 1.0], size=(n_samples, config.n_bits))
    y = np.prod(x[:, secret], axis=1)

    # Convert to GF(2): -1->0, +1->1
    A = ((x + 1) / 2).astype(np.uint8)
    b = ((y + 1) / 2).astype(np.uint8)

    n = config.n_bits
    found_secret = None

    # Try both b (odd k) and 1-b (even k)
    for b_try in [b, (1 - b).astype(np.uint8)]:
        aug = np.hstack([A.copy(), b_try.reshape(-1, 1)]).astype(np.uint8)

        pivot_cols = []
        row = 0
        for col in range(n):
            found = None
            for r in range(row, len(aug)):
                if aug[r, col] == 1:
                    found = r
                    break
            if found is None:
                continue
            aug[[row, found]] = aug[[found, row]]
            for r in range(len(aug)):
                if r != row and aug[r, col] == 1:
                    aug[r] = (aug[r] ^ aug[row])
            pivot_cols.append(col)
            row += 1

        solution = np.zeros(n, dtype=np.uint8)
        for i, col in enumerate(pivot_cols):
            solution[col] = aug[i, -1]

        candidate = sorted([i for i in range(n) if solution[i] == 1])

        # Verify on training data
        if candidate:
            y_check = np.prod(x[:, candidate], axis=1)
            if np.allclose(y_check, y):
                found_secret = candidate
                break

    if found_secret is None:
        found_secret = []

    # Verify on test data
    x_te = rng.choice([-1.0, 1.0], size=(200, config.n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    if found_secret:
        y_pred = np.prod(x_te[:, found_secret], axis=1)
        accuracy = float(np.mean(y_pred == y_te))
    else:
        accuracy = 0.0

    tracker = MemTracker()
    tracker.write("A", A.size)
    tracker.read("A")
    tracker.write("solution", n)
    s = tracker.summary()

    return {
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": found_secret,
    }


def _run_km(config, influence_samples=5, **_kwargs):
    """Kushilevitz-Mansour influence estimation via bit-flip paired queries."""
    import numpy as np

    rng = np.random.RandomState(config.seed)
    secret = sorted(rng.choice(config.n_bits, config.k_sparse, replace=False).tolist())

    rng_inf = np.random.RandomState(config.seed + 500)

    tracker = MemTracker()
    influences = np.zeros(config.n_bits)

    for i in range(config.n_bits):
        # Generate paired queries: x and x with bit i flipped
        x_batch = rng_inf.choice([-1.0, 1.0], size=(influence_samples, config.n_bits))
        tracker.write(f"x_batch_{i}", x_batch.size)

        y_orig = np.prod(x_batch[:, secret], axis=1)
        tracker.read(f"x_batch_{i}")
        tracker.write(f"y_orig_{i}", y_orig.size)

        x_flipped = x_batch.copy()
        x_flipped[:, i] *= -1
        y_flipped = np.prod(x_flipped[:, secret], axis=1)
        tracker.write(f"y_flip_{i}", y_flipped.size)

        # Influence = fraction of times label changed
        tracker.read(f"y_orig_{i}")
        tracker.read(f"y_flip_{i}")
        influences[i] = np.mean(y_orig != y_flipped)
        tracker.write(f"inf_{i}", 1)

    # Top-k bits by influence
    top_k = sorted(np.argsort(influences)[-config.k_sparse:].tolist())

    # Verify
    x_te = rng.choice([-1.0, 1.0], size=(200, config.n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    y_pred = np.prod(x_te[:, top_k], axis=1)
    accuracy = float(np.mean(y_pred == y_te))

    s = tracker.summary()
    return {
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": top_k,
    }


def _run_fourier(config, **kwargs):
    """Walsh-Hadamard / Fourier correlation."""
    import numpy as np
    from itertools import combinations

    rng = np.random.RandomState(config.seed)
    secret = sorted(rng.choice(config.n_bits, config.k_sparse, replace=False).tolist())

    n_samples = max(100, config.n_train)
    x = rng.choice([-1.0, 1.0], size=(n_samples, config.n_bits))
    y = np.prod(x[:, secret], axis=1)

    tracker = MemTracker()
    tracker.write("x", x.size)
    tracker.write("y", y.size)

    best_corr = 0
    best_subset = None

    for subset in combinations(range(config.n_bits), config.k_sparse):
        tracker.read("x")
        tracker.read("y")
        chi = np.prod(x[:, list(subset)], axis=1)
        corr = abs(np.mean(y * chi))
        if corr > best_corr:
            best_corr = corr
            best_subset = sorted(subset)

    # Verify
    x_te = rng.choice([-1.0, 1.0], size=(200, config.n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    y_pred = np.prod(x_te[:, best_subset], axis=1)
    accuracy = float(np.mean(y_pred == y_te))

    s = tracker.summary()
    return {
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": best_subset,
    }


def _run_smt(config, **kwargs):
    """Backtracking constraint solver with k-1 pruning."""
    import numpy as np
    from itertools import combinations

    rng = np.random.RandomState(config.seed)
    secret = sorted(rng.choice(config.n_bits, config.k_sparse, replace=False).tolist())

    n_samples = max(10, min(config.n_train, 20))
    x = rng.choice([-1.0, 1.0], size=(n_samples, config.n_bits))
    y = np.prod(x[:, secret], axis=1)

    tracker = MemTracker()
    tracker.write("x", x.size)
    tracker.write("y", y.size)

    # Try all k-1 subsets, check if last column is determined
    found = None
    for partial in combinations(range(config.n_bits), config.k_sparse - 1):
        tracker.read("x")
        tracker.read("y")
        partial_prod = np.prod(x[:, list(partial)], axis=1)
        residual = y * partial_prod  # should equal x[:, last_bit]

        for j in range(config.n_bits):
            if j in partial:
                continue
            if np.allclose(residual, x[:, j]):
                found = sorted(list(partial) + [j])
                break
        if found:
            break

    # Verify
    x_te = rng.choice([-1.0, 1.0], size=(200, config.n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    if found:
        y_pred = np.prod(x_te[:, found], axis=1)
        accuracy = float(np.mean(y_pred == y_te))
    else:
        accuracy = 0.0

    s = tracker.summary()
    return {
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": found,
    }


def print_result(result):
    """Print standardized output that agents can grep."""
    print(f"method: {result.get('method', 'unknown')}")
    print(f"accuracy: {result.get('accuracy', 0)}")
    if result.get('ard') is not None:
        print(f"ard: {result['ard']}")
    if result.get('dmc') is not None:
        print(f"dmc: {result['dmc']}")
    print(f"time_s: {result.get('time_s', 0)}")
    if result.get('total_floats') is not None:
        print(f"total_floats: {result['total_floats']}")
    if result.get('error'):
        print(f"error: {result['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SutroYaro evaluation harness")
    parser.add_argument("--method", required=True, help="Method to evaluate")
    parser.add_argument("--n_bits", type=int, default=20)
    parser.add_argument("--k_sparse", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--influence_samples", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    result = measure_sparse_parity(
        method=args.method,
        n_bits=args.n_bits,
        k_sparse=args.k_sparse,
        hidden=args.hidden,
        lr=args.lr,
        wd=args.wd,
        batch_size=args.batch_size,
        n_train=args.n_train,
        max_epochs=args.max_epochs,
        seed=args.seed,
        influence_samples=args.influence_samples,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_result(result)

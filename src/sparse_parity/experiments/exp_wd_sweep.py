#!/usr/bin/env python3
"""
Experiment: Weight Decay Sweep

Hypothesis: Higher weight decay accelerates grokking on 20-bit sparse parity
by encouraging simpler solutions faster.

Sweeps WD in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0] with 5 seeds each.
Adapted from fast.py (numpy-accelerated training).

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 -m sparse_parity.experiments.exp_wd_sweep
"""

import json
import time
import numpy as np
from pathlib import Path


def generate(n_bits, k_sparse, n_train, n_test, seed):
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())

    def make(n):
        x = rng.choice([-1.0, 1.0], size=(n, n_bits))
        y = np.prod(x[:, secret], axis=1)
        return x, y

    x_tr, y_tr = make(n_train)
    x_te, y_te = make(n_test)
    return x_tr, y_tr, x_te, y_te, secret


def train_one(n_bits, k_sparse, hidden, n_train, n_test, lr, wd, batch_size, max_epochs, seed):
    """Single training run. Returns (epochs_to_solve, wall_time, best_acc)."""
    x_tr, y_tr, x_te, y_te, secret = generate(n_bits, k_sparse, n_train, n_test, seed)

    rng = np.random.RandomState(seed + 1)
    std1 = np.sqrt(2.0 / n_bits)
    std2 = np.sqrt(2.0 / hidden)
    W1 = rng.randn(hidden, n_bits) * std1
    b1 = np.zeros(hidden)
    W2 = rng.randn(1, hidden) * std2
    b2 = np.zeros(1)

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(n_train)
        rng.shuffle(idx)

        for b_start in range(0, n_train, batch_size):
            b_end = min(b_start + batch_size, n_train)
            xb = x_tr[idx[b_start:b_end]]
            yb = y_tr[idx[b_start:b_end]]
            bs = xb.shape[0]

            # Forward
            h_pre = xb @ W1.T + b1
            h = np.maximum(h_pre, 0)
            out = (h @ W2.T + b2).ravel()

            # Hinge loss
            margin = out * yb
            mask = margin < 1.0
            if not np.any(mask):
                continue

            xm, ym, hm, h_pre_m = xb[mask], yb[mask], h[mask], h_pre[mask]

            dout = -ym
            dW2 = dout[:, None] * hm
            db2 = dout.sum()
            dh = dout[:, None] * W2
            dh_pre = dh * (h_pre_m > 0)
            dW1 = dh_pre.T @ xm
            db1 = dh_pre.sum(axis=0)

            W2 -= lr * (dW2.sum(axis=0, keepdims=True) / bs + wd * W2)
            b2 -= lr * (db2 / bs + wd * b2)
            W1 -= lr * (dW1 / bs + wd * W1)
            b1 -= lr * (db1 / bs + wd * b1)

        # Evaluate
        te_acc = np.mean(np.sign((np.maximum(x_te @ W1.T + b1, 0) @ W2.T + b2).ravel()) == y_te)
        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch
            break

    elapsed = time.time() - start
    return solve_epoch, elapsed, best_acc


def main():
    print("=" * 65)
    print("  WEIGHT DECAY SWEEP — 20-bit sparse parity")
    print("=" * 65)

    # Config
    n_bits, k_sparse, hidden = 20, 3, 200
    n_train, n_test = 1000, 200
    lr, batch_size, max_epochs = 0.1, 32, 200
    seeds = [42, 43, 44, 45, 46]
    wd_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    results = {}

    for wd in wd_values:
        epochs_list = []
        times_list = []
        solved = 0

        for seed in seeds:
            solve_epoch, wall_time, best_acc = train_one(
                n_bits, k_sparse, hidden, n_train, n_test,
                lr, wd, batch_size, max_epochs, seed
            )
            if solve_epoch > 0:
                solved += 1
                epochs_list.append(solve_epoch)
            times_list.append(wall_time)

        success_rate = solved / len(seeds)
        avg_epochs = sum(epochs_list) / len(epochs_list) if epochs_list else float('nan')
        avg_time = sum(times_list) / len(times_list)

        results[str(wd)] = {
            'wd': wd,
            'avg_epochs': avg_epochs if epochs_list else None,
            'avg_time_s': round(avg_time, 4),
            'success_rate': success_rate,
            'solved_count': solved,
            'total_seeds': len(seeds),
            'epochs_per_seed': epochs_list,
            'times_per_seed': [round(t, 4) for t in times_list],
        }

    # Print table
    print(f"\n{'WD':>8} | {'Avg Epochs':>10} | {'Avg Time':>10} | {'Success':>8}")
    print("-" * 48)
    for wd in wd_values:
        r = results[str(wd)]
        ep_str = f"{r['avg_epochs']:.1f}" if r['avg_epochs'] is not None else "FAIL"
        print(f"{wd:>8.3f} | {ep_str:>10} | {r['avg_time_s']:>9.3f}s | {r['success_rate']:>7.0%}")

    # Save results
    out_path = Path(__file__).resolve().parents[3] / "results" / "exp_wd_sweep" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_results = {
        'experiment': 'exp_wd_sweep',
        'hypothesis': 'Higher weight decay accelerates grokking on 20-bit sparse parity',
        'config': {
            'n_bits': n_bits, 'k_sparse': k_sparse, 'hidden': hidden,
            'n_train': n_train, 'n_test': n_test, 'lr': lr,
            'batch_size': batch_size, 'max_epochs': max_epochs,
            'seeds': seeds, 'wd_values': wd_values,
        },
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

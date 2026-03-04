#!/usr/bin/env python3
"""
Experiment: Sign SGD for k=5 sparse parity.

Hypothesis: Sign SGD needs n^{k-1} samples instead of n^k (Kou et al. 2024).
For n=20/k=5 that's 160,000 vs 3,200,000 — potentially feasible.

Key change: W -= lr * sign(grad) instead of W -= lr * grad
Weight decay applied separately (not inside sign).

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 -m sparse_parity.experiments.exp_sign_sgd
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sparse_parity.config import Config


def generate(config):
    rng = np.random.RandomState(config.seed)
    secret = sorted(rng.choice(config.n_bits, config.k_sparse, replace=False).tolist())

    def make(n):
        x = rng.choice([-1.0, 1.0], size=(n, config.n_bits))
        y = np.prod(x[:, secret], axis=1)
        return x, y

    x_tr, y_tr = make(config.n_train)
    x_te, y_te = make(config.n_test)
    return x_tr, y_tr, x_te, y_te, secret


def train(config, use_sign=True, verbose=True):
    """Training loop with optional sign SGD. Returns dict with results."""
    x_tr, y_tr, x_te, y_te, secret = generate(config)

    rng = np.random.RandomState(config.seed + 1)
    std1 = np.sqrt(2.0 / config.n_bits)
    std2 = np.sqrt(2.0 / config.hidden)
    W1 = rng.randn(config.hidden, config.n_bits) * std1
    b1 = np.zeros(config.hidden)
    W2 = rng.randn(1, config.hidden) * std2
    b2 = np.zeros(1)

    method = "sign-sgd" if use_sign else "standard"
    if verbose:
        print(f"  [{config.n_bits}-bit, k={config.k_sparse}, {method}] secret={secret}, "
              f"n_train={config.n_train}, lr={config.lr}")

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1
    epoch_90 = -1

    for epoch in range(1, config.max_epochs + 1):
        idx = np.arange(config.n_train)
        rng.shuffle(idx)

        for b_start in range(0, config.n_train, config.batch_size):
            b_end = min(b_start + config.batch_size, config.n_train)
            xb = x_tr[idx[b_start:b_end]]
            yb = y_tr[idx[b_start:b_end]]
            bs = xb.shape[0]

            # Forward
            h_pre = xb @ W1.T + b1
            h = np.maximum(h_pre, 0)
            out = (h @ W2.T + b2).ravel()

            # Hinge loss mask
            margin = out * yb
            mask = margin < 1.0
            if not np.any(mask):
                continue

            # Backward (only violated samples)
            xm = xb[mask]
            ym = yb[mask]
            hm = h[mask]
            h_pre_m = h_pre[mask]
            ms = xm.shape[0]

            dout = -ym
            dW2 = dout[:, None] * hm
            db2 = dout.sum()
            dh = dout[:, None] * W2
            dh_pre = dh * (h_pre_m > 0)
            dW1 = dh_pre.T @ xm
            db1 = dh_pre.sum(axis=0)

            if use_sign:
                # Sign SGD: sign of data gradient, weight decay applied separately
                W2 -= config.lr * (np.sign(dW2.sum(axis=0, keepdims=True) / bs) + config.wd * W2)
                b2 -= config.lr * (np.sign(db2 / bs) + config.wd * b2)
                W1 -= config.lr * (np.sign(dW1 / bs) + config.wd * W1)
                b1 -= config.lr * (np.sign(db1 / bs) + config.wd * b1)
            else:
                # Standard SGD
                W2 -= config.lr * (dW2.sum(axis=0, keepdims=True) / bs + config.wd * W2)
                b2 -= config.lr * (db2 / bs + config.wd * b2)
                W1 -= config.lr * (dW1 / bs + config.wd * W1)
                b1 -= config.lr * (db1 / bs + config.wd * b1)

        # Evaluate
        te_out = (np.maximum(x_te @ W1.T + b1, 0) @ W2.T + b2).ravel()
        te_acc = np.mean(np.sign(te_out) == y_te)
        tr_out = (np.maximum(x_tr @ W1.T + b1, 0) @ W2.T + b2).ravel()
        tr_acc = np.mean(np.sign(tr_out) == y_tr)

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= 0.90 and epoch_90 < 0:
            epoch_90 = epoch
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch

        if verbose and (epoch % 20 == 0 or epoch == 1 or te_acc >= 0.90):
            print(f"    epoch {epoch:>4}: train={tr_acc:.0%} test={te_acc:.0%}")

        if te_acc >= 1.0:
            break

    elapsed = time.time() - start
    if verbose:
        print(f"  Result: {best_acc:.0%} in {elapsed:.2f}s ({epoch} epochs)")

    return {
        'method': method,
        'best_test_acc': round(float(best_acc), 4),
        'solve_epoch': solve_epoch,
        'epoch_90': epoch_90,
        'total_epochs': epoch,
        'elapsed_s': round(elapsed, 3),
        'secret': secret,
        'n_bits': config.n_bits,
        'k_sparse': config.k_sparse,
        'n_train': config.n_train,
        'hidden': config.hidden,
        'lr': config.lr,
        'wd': config.wd,
        'batch_size': config.batch_size,
        'max_epochs': config.max_epochs,
    }


def run_config(n_bits, k_sparse, n_train, hidden, lr, max_epochs, seeds, use_sign, verbose=True):
    """Run multiple seeds for one config. Returns list of results."""
    results = []
    for seed in seeds:
        config = Config(
            n_bits=n_bits, k_sparse=k_sparse, hidden=hidden,
            lr=lr, wd=0.01, max_epochs=max_epochs,
            n_train=n_train, n_test=500, seed=seed,
        )
        config.batch_size = 32
        r = train(config, use_sign=use_sign, verbose=(verbose and seed == seeds[0]))
        results.append(r)
        if not verbose or seed != seeds[0]:
            status = "SOLVED" if r['best_test_acc'] >= 0.95 else f"{r['best_test_acc']:.0%}"
            print(f"    seed={seed}: {r['elapsed_s']:.2f}s  {status}  (ep90={r['epoch_90']})")
    return results


def main():
    print("=" * 70)
    print("  EXPERIMENT: Sign SGD for Sparse Parity")
    print("  Hypothesis: sign(grad) needs n^{k-1} vs n^k samples")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # -----------------------------------------------------------------------
    # 1) Sanity: n=20, k=3 — Sign SGD vs Standard SGD
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=20, k=3 (sanity check)")
    print("=" * 70)

    print("\n  --- Standard SGD (baseline) ---")
    std_20_3 = run_config(20, 3, n_train=1000, hidden=200, lr=0.1,
                          max_epochs=200, seeds=seeds, use_sign=False)
    all_results['n20_k3_standard'] = std_20_3

    print("\n  --- Sign SGD ---")
    sign_20_3 = run_config(20, 3, n_train=1000, hidden=200, lr=0.01,
                           max_epochs=200, seeds=seeds, use_sign=True)
    all_results['n20_k3_sign'] = sign_20_3

    # Also try lr=0.001 for sign SGD
    print("\n  --- Sign SGD (lr=0.001) ---")
    sign_20_3_b = run_config(20, 3, n_train=1000, hidden=200, lr=0.001,
                             max_epochs=200, seeds=seeds, use_sign=True)
    all_results['n20_k3_sign_lr001'] = sign_20_3_b

    # -----------------------------------------------------------------------
    # 2) THE KEY TEST: n=20, k=5
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=20, k=5 (the hard one — standard SGD gets 61.5%)")
    print("  Theory: sign SGD needs n^{k-1} = 20^4 = 160,000 samples")
    print("=" * 70)

    # Standard SGD baseline (should fail ~61%)
    print("\n  --- Standard SGD (baseline, expected ~61%) ---")
    std_20_5 = run_config(20, 5, n_train=5000, hidden=500, lr=0.1,
                          max_epochs=500, seeds=seeds, use_sign=False)
    all_results['n20_k5_standard'] = std_20_5

    # Sign SGD — try several configs
    # Start with moderate n_train, increase if needed
    for n_train, lr, label in [
        (5000, 0.01, "5K samples"),
        (5000, 0.001, "5K samples, lr=0.001"),
        (20000, 0.01, "20K samples"),
        (20000, 0.001, "20K samples, lr=0.001"),
        (50000, 0.001, "50K samples"),
    ]:
        print(f"\n  --- Sign SGD ({label}) ---")
        r = run_config(20, 5, n_train=n_train, hidden=500, lr=lr,
                       max_epochs=500, seeds=seeds, use_sign=True)
        key = f'n20_k5_sign_n{n_train}_lr{lr}'
        all_results[key] = r

    # -----------------------------------------------------------------------
    # 3) n=30, k=3
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 3: n=30, k=3")
    print("=" * 70)

    print("\n  --- Standard SGD ---")
    std_30_3 = run_config(30, 3, n_train=2000, hidden=200, lr=0.1,
                          max_epochs=300, seeds=seeds, use_sign=False)
    all_results['n30_k3_standard'] = std_30_3

    print("\n  --- Sign SGD ---")
    sign_30_3 = run_config(30, 3, n_train=2000, hidden=200, lr=0.01,
                           max_epochs=300, seeds=seeds, use_sign=True)
    all_results['n30_k3_sign'] = sign_30_3

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  COMPARISON TABLE")
    print("=" * 90)
    header = f"{'Config':<35} | {'Best Acc':>8} | {'Ep->90%':>8} | {'Time(s)':>8} | {'Solved':>6}"
    print(header)
    print("-" * 90)

    for key, runs in all_results.items():
        avg_acc = np.mean([r['best_test_acc'] for r in runs])
        avg_ep90 = np.mean([r['epoch_90'] for r in runs if r['epoch_90'] > 0])
        avg_time = np.mean([r['elapsed_s'] for r in runs])
        n_solved = sum(1 for r in runs if r['best_test_acc'] >= 0.95)
        ep90_str = f"{avg_ep90:.0f}" if not np.isnan(avg_ep90) else "---"
        print(f"  {key:<33} | {avg_acc:>8.1%} | {ep90_str:>8} | {avg_time:>8.2f} | {n_solved}/{len(runs):>4}")

    print("=" * 90)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_sign_sgd'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Serialize
    serializable = {}
    for key, runs in all_results.items():
        serializable[key] = runs

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_sign_sgd',
            'description': 'Sign SGD vs standard SGD on sparse parity (k=3 and k=5)',
            'hypothesis': 'Sign SGD needs n^{k-1} samples vs n^k, enabling k=5',
            'configs': serializable,
        }, f, indent=2)

    print(f"\n  Results saved to: {results_path}")

    return all_results


if __name__ == '__main__':
    main()

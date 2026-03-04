#!/usr/bin/env python3
"""
Fast sparse parity training — numpy-accelerated.

Target: full 20-bit solve in <2 seconds.
Same algorithm as train.py, just using numpy for matrix ops.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 -m sparse_parity.fast
"""

import time
import numpy as np
from .config import Config


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


def train(config, verbose=True):
    """Full training loop. Returns dict with results."""
    x_tr, y_tr, x_te, y_te, secret = generate(config)

    rng = np.random.RandomState(config.seed + 1)
    std1 = np.sqrt(2.0 / config.n_bits)
    std2 = np.sqrt(2.0 / config.hidden)
    W1 = rng.randn(config.hidden, config.n_bits) * std1
    b1 = np.zeros(config.hidden)
    W2 = rng.randn(1, config.hidden) * std2
    b2 = np.zeros(1)

    if verbose:
        print(f"  [{config.n_bits}-bit, k={config.k_sparse}] secret={secret}, "
              f"params={config.hidden * config.n_bits + config.hidden + config.hidden + 1:,}")

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1

    for epoch in range(1, config.max_epochs + 1):
        # Mini-batch SGD
        idx = np.arange(config.n_train)
        rng.shuffle(idx)

        for b_start in range(0, config.n_train, config.batch_size):
            b_end = min(b_start + config.batch_size, config.n_train)
            xb = x_tr[idx[b_start:b_end]]
            yb = y_tr[idx[b_start:b_end]]
            bs = xb.shape[0]

            # Forward
            h_pre = xb @ W1.T + b1          # (bs, hidden)
            h = np.maximum(h_pre, 0)         # ReLU
            out = h @ W2.T + b2              # (bs, 1)
            out = out.ravel()                # (bs,)

            # Hinge loss mask
            margin = out * yb
            mask = margin < 1.0
            if not np.any(mask):
                continue

            # Backward (only on violated samples)
            xm = xb[mask]
            ym = yb[mask]
            hm = h[mask]
            h_pre_m = h_pre[mask]
            ms = xm.shape[0]

            dout = -ym                           # (ms,)
            dW2 = dout[:, None] * hm             # (ms, hidden)
            db2 = dout.sum()
            dh = dout[:, None] * W2              # (ms, hidden)
            dh_pre = dh * (h_pre_m > 0)          # ReLU backward
            dW1 = dh_pre.T @ xm                  # (hidden, n_bits)
            db1 = dh_pre.sum(axis=0)

            # SGD update (averaged over batch)
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
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch

        if verbose and (epoch % 5 == 0 or epoch == 1 or te_acc >= 0.95):
            print(f"    epoch {epoch:>3}: train={tr_acc:.0%} test={te_acc:.0%}")

        if te_acc >= 1.0:
            break

    elapsed = time.time() - start

    if verbose:
        print(f"  Result: {best_acc:.0%} in {elapsed:.2f}s ({epoch} epochs)")

    return {
        'best_test_acc': best_acc,
        'solve_epoch': solve_epoch,
        'total_epochs': epoch,
        'elapsed_s': elapsed,
        'secret': secret,
        'config': {k: v for k, v in config.__dict__.items()},
    }


def main():
    print("=" * 60)
    print("  FAST SPARSE PARITY (numpy)")
    print("=" * 60)

    # Fast 20-bit solve — under 0.2s
    config = Config(
        n_bits=20, k_sparse=3, hidden=200,
        lr=0.1, wd=0.01, max_epochs=200,
        n_train=1000, n_test=200, seed=42,
    )
    config.batch_size = 32

    # Run 5 seeds to show it's robust
    times = []
    for seed in [42, 43, 44, 45, 46]:
        config.seed = seed
        r = train(config, verbose=(seed == 42))
        times.append(r['elapsed_s'])
        if seed != 42:
            status = "SOLVED" if r['best_test_acc'] >= 0.95 else f"{r['best_test_acc']:.0%}"
            print(f"  seed={seed}: {r['elapsed_s']:.2f}s  {status}  (epoch {r['solve_epoch']})")

    print(f"\n  Avg: {sum(times)/len(times):.2f}s  Min: {min(times):.2f}s  Max: {max(times):.2f}s")


if __name__ == '__main__':
    main()

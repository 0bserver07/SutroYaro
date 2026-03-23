#!/usr/bin/env python3
"""
Experiment: GrokFast + Curriculum — Do they compound?

Hypothesis: Curriculum learning neutralizes the n-scaling wall (14.6x on n=50/k=3).
GrokFast accelerates the k-th order grokking plateau (2.5x on n=20/k=5).
They attack different axes, so combining them should compound the gains,
especially on hard configs like n=50/k=5 where both n and k are large.

Answers: "Can GrokFast + curriculum compound?" (TODO.md, exp_grokfast_v2 open question)

Usage:
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_grokfast_curriculum.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# =============================================================================
# CONFIG
# =============================================================================

EXP_NAME = "exp_grokfast_curriculum"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / EXP_NAME

LR = 0.1
WD = 0.01
BATCH_SIZE = 32
HIDDEN = 200
N_TRAIN = 2000
N_TEST = 200

SEEDS = [42, 43, 44, 45, 46]

# GrokFast setting: aggressive (best on k=5 from exp_grokfast_v2)
GF_ALPHA = 0.98
GF_LAM = 2.0


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def generate_data(n_bits, k_sparse, secret, n_samples, rng):
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    return x, y


def init_weights(hidden, n_bits, seed=42):
    rng = np.random.RandomState(seed)
    std1 = np.sqrt(2.0 / n_bits)
    std2 = np.sqrt(2.0 / hidden)
    W1 = rng.randn(hidden, n_bits) * std1
    b1 = np.zeros(hidden)
    W2 = rng.randn(1, hidden) * std2
    b2 = np.zeros(1)
    return W1, b1, W2, b2


def expand_W1(W1, old_n, new_n, rng_seed=99):
    hidden = W1.shape[0]
    rng = np.random.RandomState(rng_seed)
    new_std = np.sqrt(2.0 / new_n) * 0.1
    W1_new = np.zeros((hidden, new_n))
    W1_new[:, :old_n] = W1
    W1_new[:, old_n:] = rng.randn(hidden, new_n - old_n) * new_std
    return W1_new


def train_phase(W1, b1, W2, b2, n_bits, k_sparse, secret,
                max_epochs=500, target_acc=0.95, rng_seed=42,
                grokfast=False, alpha=0.98, lam=2.0):
    """Train one phase. Returns (W1, b1, W2, b2, result_dict)."""
    rng = np.random.RandomState(rng_seed)
    x_tr, y_tr = generate_data(n_bits, k_sparse, secret, N_TRAIN, rng)
    x_te, y_te = generate_data(n_bits, k_sparse, secret, N_TEST, rng)

    # EMA buffers for GrokFast
    if grokfast:
        ema_W1 = np.zeros_like(W1)
        ema_b1 = np.zeros_like(b1)
        ema_W2 = np.zeros_like(W2)
        ema_b2 = np.zeros_like(b2)

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(N_TRAIN)
        rng.shuffle(idx)

        for b_start in range(0, N_TRAIN, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, N_TRAIN)
            xb = x_tr[idx[b_start:b_end]]
            yb = y_tr[idx[b_start:b_end]]
            bs = xb.shape[0]

            h_pre = xb @ W1.T + b1
            h = np.maximum(h_pre, 0)
            out = (h @ W2.T + b2).ravel()

            margin = out * yb
            mask = margin < 1.0
            if not np.any(mask):
                continue

            xm, ym, hm, h_pre_m = xb[mask], yb[mask], h[mask], h_pre[mask]

            dout = -ym
            gW2 = (dout[:, None] * hm).sum(axis=0, keepdims=True) / bs
            gb2 = dout.sum() / bs
            dh = dout[:, None] * W2
            dh_pre = dh * (h_pre_m > 0)
            gW1 = (dh_pre.T @ xm) / bs
            gb1 = dh_pre.sum(axis=0) / bs

            if grokfast:
                ema_W1 = alpha * ema_W1 + (1 - alpha) * gW1
                ema_b1 = alpha * ema_b1 + (1 - alpha) * gb1
                ema_W2 = alpha * ema_W2 + (1 - alpha) * gW2
                ema_b2 = alpha * ema_b2 + (1 - alpha) * gb2
                gW1 = gW1 + lam * ema_W1
                gb1 = gb1 + lam * ema_b1
                gW2 = gW2 + lam * ema_W2
                gb2 = gb2 + lam * ema_b2

            W2 -= LR * (gW2 + WD * W2)
            b2 -= LR * (gb2 + WD * b2)
            W1 -= LR * (gW1 + WD * W1)
            b1 -= LR * (gb1 + WD * b1)

        te_out = (np.maximum(x_te @ W1.T + b1, 0) @ W2.T + b2).ravel()
        te_acc = float(np.mean(np.sign(te_out) == y_te))

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= target_acc and solve_epoch < 0:
            solve_epoch = epoch
            break

    elapsed = time.time() - start
    return W1, b1, W2, b2, {
        'best_test_acc': best_acc,
        'solve_epoch': solve_epoch,
        'total_epochs': epoch,
        'elapsed_s': round(elapsed, 4),
    }


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_direct(n_bits, k_sparse, secret, seed, max_epochs=500, grokfast=False):
    """Direct training (no curriculum)."""
    W1, b1, W2, b2 = init_weights(HIDDEN, n_bits, seed=seed + 1)
    _, _, _, _, result = train_phase(
        W1, b1, W2, b2, n_bits, k_sparse, secret,
        max_epochs=max_epochs, rng_seed=seed,
        grokfast=grokfast, alpha=GF_ALPHA, lam=GF_LAM,
    )
    return result


def run_n_curriculum(stages, k_sparse, secret, seed, max_epochs_per_phase=500,
                     grokfast=False):
    """n-curriculum: train on small n, expand, repeat."""
    total_start = time.time()
    total_epochs = 0

    n0 = stages[0]
    W1, b1, W2, b2 = init_weights(HIDDEN, n0, seed=seed + 1)
    W1, b1, W2, b2, phase_r = train_phase(
        W1, b1, W2, b2, n0, k_sparse, secret,
        max_epochs=max_epochs_per_phase, rng_seed=seed,
        grokfast=grokfast, alpha=GF_ALPHA, lam=GF_LAM,
    )
    total_epochs += phase_r['total_epochs']

    for i, n_next in enumerate(stages[1:], start=2):
        n_prev = stages[i - 2]
        W1 = expand_W1(W1, n_prev, n_next, rng_seed=seed + i * 100)
        W1, b1, W2, b2, phase_r = train_phase(
            W1, b1, W2, b2, n_next, k_sparse, secret,
            max_epochs=max_epochs_per_phase, rng_seed=seed + i * 1000,
            grokfast=grokfast, alpha=GF_ALPHA, lam=GF_LAM,
        )
        total_epochs += phase_r['total_epochs']

    total_elapsed = time.time() - total_start
    return {
        'best_test_acc': phase_r['best_test_acc'],
        'total_epochs': total_epochs,
        'elapsed_s': round(total_elapsed, 4),
    }


def run_multi_seed(runner, seeds, label, **kwargs):
    """Run a method across multiple seeds, print and return summary."""
    accs, epochs, times = [], [], []
    for seed in seeds:
        r = runner(seed=seed, **kwargs)
        accs.append(r['best_test_acc'])
        ep = r.get('solve_epoch', r['total_epochs'])
        if ep < 0:
            ep = r['total_epochs']
        epochs.append(ep)
        times.append(r['elapsed_s'])

    solve_rate = sum(1 for a in accs if a >= 0.95) / len(seeds)
    avg_ep = sum(epochs) / len(epochs)
    avg_t = sum(times) / len(times)
    print(f"  {label:<40} solve={solve_rate:.0%}  epochs={avg_ep:>6.0f}  time={avg_t:.3f}s")

    return {
        'label': label,
        'solve_rate': solve_rate,
        'avg_epochs': avg_ep,
        'avg_time': avg_t,
        'accs': accs,
        'epochs': epochs,
        'times': times,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  EXPERIMENT: GrokFast + Curriculum — Compounding Test")
    print("  4 methods x 3 regimes x 5 seeds = 60 runs")
    print("=" * 78)

    all_results = {}

    # --- Regime 1: n=20, k=5 (GrokFast's strength) ---
    print(f"\n{'─'*78}")
    print(f"  REGIME: n=20, k=5 (high interaction order)")
    print(f"{'─'*78}")

    rng = np.random.RandomState(42)
    secret_k5_20 = sorted(rng.choice(10, 5, replace=False).tolist())  # all < 10 for curriculum
    print(f"  Secret: {secret_k5_20}\n")

    r1 = {}
    r1['sgd'] = run_multi_seed(
        run_direct, SEEDS, "SGD direct",
        n_bits=20, k_sparse=5, secret=secret_k5_20, max_epochs=500)
    r1['grokfast'] = run_multi_seed(
        run_direct, SEEDS, "GrokFast direct",
        n_bits=20, k_sparse=5, secret=secret_k5_20, max_epochs=500, grokfast=True)
    r1['curriculum'] = run_multi_seed(
        run_n_curriculum, SEEDS, "Curriculum [10->20]",
        stages=[10, 20], k_sparse=5, secret=secret_k5_20)
    r1['grokfast_curriculum'] = run_multi_seed(
        run_n_curriculum, SEEDS, "GrokFast + Curriculum [10->20]",
        stages=[10, 20], k_sparse=5, secret=secret_k5_20, grokfast=True)
    all_results['n20_k5'] = r1

    # --- Regime 2: n=50, k=3 (Curriculum's strength) ---
    print(f"\n{'─'*78}")
    print(f"  REGIME: n=50, k=3 (high input dimension)")
    print(f"{'─'*78}")

    secret_k3_50 = sorted(rng.choice(10, 3, replace=False).tolist())  # all < 10
    print(f"  Secret: {secret_k3_50}\n")

    r2 = {}
    r2['sgd'] = run_multi_seed(
        run_direct, SEEDS, "SGD direct",
        n_bits=50, k_sparse=3, secret=secret_k3_50, max_epochs=500)
    r2['grokfast'] = run_multi_seed(
        run_direct, SEEDS, "GrokFast direct",
        n_bits=50, k_sparse=3, secret=secret_k3_50, max_epochs=500, grokfast=True)
    r2['curriculum'] = run_multi_seed(
        run_n_curriculum, SEEDS, "Curriculum [10->30->50]",
        stages=[10, 30, 50], k_sparse=3, secret=secret_k3_50)
    r2['grokfast_curriculum'] = run_multi_seed(
        run_n_curriculum, SEEDS, "GrokFast + Curriculum [10->30->50]",
        stages=[10, 30, 50], k_sparse=3, secret=secret_k3_50, grokfast=True)
    all_results['n50_k3'] = r2

    # --- Regime 3: n=50, k=5 (both hard) ---
    print(f"\n{'─'*78}")
    print(f"  REGIME: n=50, k=5 (both hard — the real test)")
    print(f"{'─'*78}")

    secret_k5_50 = sorted(rng.choice(10, 5, replace=False).tolist())  # all < 10
    print(f"  Secret: {secret_k5_50}\n")

    r3 = {}
    r3['sgd'] = run_multi_seed(
        run_direct, SEEDS, "SGD direct",
        n_bits=50, k_sparse=5, secret=secret_k5_50, max_epochs=1000)
    r3['grokfast'] = run_multi_seed(
        run_direct, SEEDS, "GrokFast direct",
        n_bits=50, k_sparse=5, secret=secret_k5_50, max_epochs=1000, grokfast=True)
    r3['curriculum'] = run_multi_seed(
        run_n_curriculum, SEEDS, "Curriculum [10->30->50]",
        stages=[10, 30, 50], k_sparse=5, secret=secret_k5_50, max_epochs_per_phase=500)
    r3['grokfast_curriculum'] = run_multi_seed(
        run_n_curriculum, SEEDS, "GrokFast + Curriculum [10->30->50]",
        stages=[10, 30, 50], k_sparse=5, secret=secret_k5_50, max_epochs_per_phase=500,
        grokfast=True)
    all_results['n50_k5'] = r3

    # --- Summary ---
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"  {'Regime':<12} {'Method':<40} {'Solve%':>7} {'Epochs':>8} {'Time':>9}")
    print(f"  {'─'*12} {'─'*40} {'─'*7} {'─'*8} {'─'*9}")

    for regime_name, regime_results in all_results.items():
        for method_name, r in regime_results.items():
            print(f"  {regime_name:<12} {r['label']:<40} {r['solve_rate']:>6.0%} "
                  f"{r['avg_epochs']:>8.0f} {r['avg_time']:>8.3f}s")
        print()

    # --- Save ---
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {RESULTS_DIR / 'results.json'}")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()

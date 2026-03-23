#!/usr/bin/env python3
"""
Experiment: GrokFast v2 — Testing on harder regimes where grokking plateau is long.

Hypothesis: GrokFast (EMA gradient amplification) helps when the grokking plateau is
genuinely extended (large n, large k), even though it was counterproductive on the
easy n=20/k=3 regime where SGD converges in 5 epochs (exp4).

Answers: "Can GrokFast accelerate grokking on harder sparse parity configurations?"
Reference: Lee et al. 2024, https://arxiv.org/abs/2405.20233

Usage:
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_grokfast_v2.py
"""

import time
import json
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sparse_parity.config import Config
from sparse_parity.tracker import MemTracker

# =============================================================================
# CONFIG
# =============================================================================

EXP_NAME = "exp_grokfast_v2"
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / EXP_NAME

# Test configs: easy (baseline confirmation) + hard regimes
CONFIGS = {
    "easy_n20_k3": Config(
        n_bits=20, k_sparse=3, hidden=200,
        lr=0.1, wd=0.01, max_epochs=200,
        n_train=1000, n_test=200, seed=42,
    ),
    "hard_n30_k3": Config(
        n_bits=30, k_sparse=3, hidden=200,
        lr=0.1, wd=0.01, max_epochs=200,
        n_train=1000, n_test=200, seed=42,
    ),
    "hard_n20_k5": Config(
        n_bits=20, k_sparse=5, hidden=200,
        lr=0.1, wd=0.01, max_epochs=500,
        n_train=2000, n_test=200, seed=42,
    ),
}

# GrokFast hyperparameter grid
GROKFAST_PARAMS = [
    {"alpha": 0.98, "lam": 2.0},   # original paper defaults
    {"alpha": 0.95, "lam": 1.0},   # less aggressive
    {"alpha": 0.99, "lam": 0.5},   # gentle amplification
]

SEEDS = [42, 43, 44, 45, 46]

# Set batch_size for all configs
for c in CONFIGS.values():
    c.batch_size = 32


# =============================================================================
# DATA GENERATION
# =============================================================================

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


# =============================================================================
# TRAINING: SGD BASELINE (numpy)
# =============================================================================

def train_sgd(config, verbose=False):
    x_tr, y_tr, x_te, y_te, secret = generate(config)

    rng = np.random.RandomState(config.seed + 1)
    std1 = np.sqrt(2.0 / config.n_bits)
    std2 = np.sqrt(2.0 / config.hidden)
    W1 = rng.randn(config.hidden, config.n_bits) * std1
    b1 = np.zeros(config.hidden)
    W2 = rng.randn(1, config.hidden) * std2
    b2 = np.zeros(1)

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1

    for epoch in range(1, config.max_epochs + 1):
        idx = np.arange(config.n_train)
        rng.shuffle(idx)

        for b_start in range(0, config.n_train, config.batch_size):
            b_end = min(b_start + config.batch_size, config.n_train)
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
            ms = xm.shape[0]

            dout = -ym
            dW2 = dout[:, None] * hm
            db2 = dout.sum()
            dh = dout[:, None] * W2
            dh_pre = dh * (h_pre_m > 0)
            dW1 = dh_pre.T @ xm
            db1_g = dh_pre.sum(axis=0)

            W2 -= config.lr * (dW2.sum(axis=0, keepdims=True) / bs + config.wd * W2)
            b2 -= config.lr * (db2 / bs + config.wd * b2)
            W1 -= config.lr * (dW1 / bs + config.wd * W1)
            b1 -= config.lr * (db1_g / bs + config.wd * b1)

        te_out = (np.maximum(x_te @ W1.T + b1, 0) @ W2.T + b2).ravel()
        te_acc = float(np.mean(np.sign(te_out) == y_te))

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch
            break

    elapsed = time.time() - start
    return {
        "method": "sgd",
        "best_test_acc": best_acc,
        "solve_epoch": solve_epoch,
        "total_epochs": epoch,
        "elapsed_s": elapsed,
    }


# =============================================================================
# TRAINING: GROKFAST (numpy)
# =============================================================================

def train_grokfast(config, alpha=0.98, lam=2.0, verbose=False):
    x_tr, y_tr, x_te, y_te, secret = generate(config)

    rng = np.random.RandomState(config.seed + 1)
    std1 = np.sqrt(2.0 / config.n_bits)
    std2 = np.sqrt(2.0 / config.hidden)
    W1 = rng.randn(config.hidden, config.n_bits) * std1
    b1 = np.zeros(config.hidden)
    W2 = rng.randn(1, config.hidden) * std2
    b2 = np.zeros(1)

    # EMA buffers
    ema_W1 = np.zeros_like(W1)
    ema_b1 = np.zeros_like(b1)
    ema_W2 = np.zeros_like(W2)
    ema_b2 = np.zeros_like(b2)

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1

    for epoch in range(1, config.max_epochs + 1):
        idx = np.arange(config.n_train)
        rng.shuffle(idx)

        for b_start in range(0, config.n_train, config.batch_size):
            b_end = min(b_start + config.batch_size, config.n_train)
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
            ms = xm.shape[0]

            dout = -ym
            dW2 = dout[:, None] * hm
            db2 = dout.sum()
            dh = dout[:, None] * W2
            dh_pre = dh * (h_pre_m > 0)
            dW1 = dh_pre.T @ xm
            db1_g = dh_pre.sum(axis=0)

            # Average gradients
            gW2 = dW2.sum(axis=0, keepdims=True) / bs
            gb2 = db2 / bs
            gW1 = dW1 / bs
            gb1 = db1_g / bs

            # GrokFast: EMA filter + amplification
            ema_W1 = alpha * ema_W1 + (1 - alpha) * gW1
            ema_b1 = alpha * ema_b1 + (1 - alpha) * gb1
            ema_W2 = alpha * ema_W2 + (1 - alpha) * gW2
            ema_b2 = alpha * ema_b2 + (1 - alpha) * gb2

            gW1_f = gW1 + lam * ema_W1
            gb1_f = gb1 + lam * ema_b1
            gW2_f = gW2 + lam * ema_W2
            gb2_f = gb2 + lam * ema_b2

            # SGD update with filtered gradients
            W2 -= config.lr * (gW2_f + config.wd * W2)
            b2 -= config.lr * (gb2_f + config.wd * b2)
            W1 -= config.lr * (gW1_f + config.wd * W1)
            b1 -= config.lr * (gb1_f + config.wd * b1)

        te_out = (np.maximum(x_te @ W1.T + b1, 0) @ W2.T + b2).ravel()
        te_acc = float(np.mean(np.sign(te_out) == y_te))

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch
            break

    elapsed = time.time() - start
    return {
        "method": f"grokfast(a={alpha},l={lam})",
        "alpha": alpha,
        "lam": lam,
        "best_test_acc": best_acc,
        "solve_epoch": solve_epoch,
        "total_epochs": epoch,
        "elapsed_s": elapsed,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  EXPERIMENT: GrokFast v2 — Hard Regime Testing")
    print("  Hypothesis: GrokFast helps when grokking plateau is genuinely long")
    print("=" * 78)

    all_results = {}

    for regime_name, base_config in CONFIGS.items():
        print(f"\n{'─'*78}")
        print(f"  REGIME: {regime_name}  (n={base_config.n_bits}, k={base_config.k_sparse}, "
              f"max_epochs={base_config.max_epochs})")
        print(f"{'─'*78}")

        regime_results = []

        # --- SGD baseline across seeds ---
        sgd_accs, sgd_epochs, sgd_times = [], [], []
        for seed in SEEDS:
            cfg = Config(**{k: v for k, v in base_config.__dict__.items()})
            cfg.batch_size = base_config.batch_size
            cfg.seed = seed
            r = train_sgd(cfg)
            sgd_accs.append(r["best_test_acc"])
            sgd_epochs.append(r["solve_epoch"] if r["solve_epoch"] > 0 else r["total_epochs"])
            sgd_times.append(r["elapsed_s"])
            regime_results.append({**r, "seed": seed, "regime": regime_name})

        solve_rate = sum(1 for a in sgd_accs if a >= 0.95) / len(SEEDS)
        avg_time = sum(sgd_times) / len(sgd_times)
        avg_epoch = sum(sgd_epochs) / len(sgd_epochs)
        print(f"\n  SGD baseline:  solve={solve_rate:.0%}  avg_epoch={avg_epoch:.0f}  "
              f"avg_time={avg_time:.2f}s  accs={[f'{a:.0%}' for a in sgd_accs]}")

        # --- GrokFast variants across seeds ---
        for gf_params in GROKFAST_PARAMS:
            gf_accs, gf_epochs, gf_times = [], [], []
            for seed in SEEDS:
                cfg = Config(**{k: v for k, v in base_config.__dict__.items()})
                cfg.batch_size = base_config.batch_size
                cfg.seed = seed
                r = train_grokfast(cfg, **gf_params)
                gf_accs.append(r["best_test_acc"])
                gf_epochs.append(r["solve_epoch"] if r["solve_epoch"] > 0 else r["total_epochs"])
                gf_times.append(r["elapsed_s"])
                regime_results.append({**r, "seed": seed, "regime": regime_name})

            solve_rate = sum(1 for a in gf_accs if a >= 0.95) / len(SEEDS)
            avg_time = sum(gf_times) / len(gf_times)
            avg_epoch = sum(gf_epochs) / len(gf_epochs)
            label = f"a={gf_params['alpha']},l={gf_params['lam']}"
            print(f"  GrokFast({label}): solve={solve_rate:.0%}  avg_epoch={avg_epoch:.0f}  "
                  f"avg_time={avg_time:.2f}s  accs={[f'{a:.0%}' for a in gf_accs]}")

        all_results[regime_name] = regime_results

    # --- Summary table ---
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"  {'Regime':<16} {'Method':<28} {'Solve%':>7} {'AvgEpoch':>9} {'AvgTime':>9}")
    print(f"  {'─'*16} {'─'*28} {'─'*7} {'─'*9} {'─'*9}")

    for regime_name, results in all_results.items():
        # Group by method
        methods = {}
        for r in results:
            m = r["method"]
            if m not in methods:
                methods[m] = []
            methods[m].append(r)

        for method, runs in methods.items():
            accs = [r["best_test_acc"] for r in runs]
            epochs = [r["solve_epoch"] if r["solve_epoch"] > 0 else r["total_epochs"] for r in runs]
            times = [r["elapsed_s"] for r in runs]
            solve_pct = sum(1 for a in accs if a >= 0.95) / len(runs)
            print(f"  {regime_name:<16} {method:<28} {solve_pct:>6.0%} {sum(epochs)/len(epochs):>9.0f} "
                  f"{sum(times)/len(times):>8.2f}s")

    # --- Save ---
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_DIR / 'results.json'}")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()

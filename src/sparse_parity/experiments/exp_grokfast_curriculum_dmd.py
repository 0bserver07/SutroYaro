#!/usr/bin/env python3
"""
Experiment: GrokFast + Curriculum DMD measurement

Measures DMD (Data Movement Distance) for GrokFast + Curriculum methods using
buffer-level LRU stack tracking. Runs one tracked SGD step per method to get
per-step DMD, then estimates total DMD from convergence data.

Uses LRUStackTracker at buffer granularity (one stack entry per buffer, cost
weighted by size). Per-element tracking is too slow for these array sizes
(O(n^2) for n=4000 float arrays); splay tree optimization would fix this
but isn't implemented yet.

Answers: Yad's review feedback on PR #51 ("no DMC/ARD measurement").

Usage:
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_grokfast_curriculum_dmd.py
"""

import sys
import time
import json
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# =============================================================================
# CONFIG
# =============================================================================

EXP_NAME = "exp_grokfast_curriculum_dmd"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / EXP_NAME

LR = 0.1
WD = 0.01
BATCH_SIZE = 32
HIDDEN = 200
N_TRAIN = 2000
N_TEST = 200

GF_ALPHA = 0.98
GF_LAM = 2.0

SEED = 42


# =============================================================================
# BUFFER-LEVEL LRU TRACKER
# =============================================================================

class BufferLRUTracker:
    """LRU stack tracker at buffer granularity.

    Each named buffer is one entry in the LRU stack. DMD per read =
    size * sqrt(stack_position), where stack_position is the 1-indexed
    position in the LRU order (1 = most recently written).

    Writes move the buffer to top of stack (free, no DMD cost).
    Reads observe position without moving (pay DMD).
    """

    def __init__(self):
        self._stack = []       # most recent at index 0
        self._pos = {}         # name -> index
        self._sizes = {}       # name -> size
        self._dmd = 0.0
        self._reads = 0
        self._writes = 0

    def write(self, name, size):
        self._sizes[name] = size
        if name in self._pos:
            idx = self._pos[name]
            self._stack.pop(idx)
        self._stack.insert(0, name)
        # Rebuild position index
        for i, n in enumerate(self._stack):
            self._pos[n] = i
        self._writes += 1

    def read(self, name, size=None):
        if size is None:
            size = self._sizes.get(name, 0)
        if name in self._pos:
            dist = self._pos[name] + 1  # 1-indexed
        else:
            dist = len(self._stack) + 1
        cost = size * math.sqrt(dist)
        self._dmd += cost
        self._reads += 1
        return dist, cost

    def summary(self):
        return {
            'dmd': self._dmd,
            'reads': self._reads,
            'writes': self._writes,
            'stack_size': len(self._stack),
        }


# =============================================================================
# DATA
# =============================================================================

def generate_data(n_bits, k_sparse, secret, n_samples, rng):
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    return x, y


# =============================================================================
# ONE TRACKED SGD STEP (buffer-level DMD)
# =============================================================================

def tracked_sgd_step(n_bits, k_sparse, secret, seed,
                     grokfast=False, alpha=0.98, lam=2.0):
    """Run one mini-batch SGD step with buffer-level DMD tracking.

    Mirrors the exact read/write sequence of train_phase() from
    exp_grokfast_curriculum.py, but records each buffer access.
    """
    tracker = BufferLRUTracker()

    rng = np.random.RandomState(seed)
    x_all, y_all = generate_data(n_bits, k_sparse, secret, N_TRAIN, rng)

    rng2 = np.random.RandomState(seed + 1)
    std1 = np.sqrt(2.0 / n_bits)
    std2 = np.sqrt(2.0 / HIDDEN)
    W1 = rng2.randn(HIDDEN, n_bits) * std1
    b1 = np.zeros(HIDDEN)
    W2 = rng2.randn(1, HIDDEN) * std2
    b2 = np.zeros(1)

    # Write initial params + data
    tracker.write('W1', HIDDEN * n_bits)
    tracker.write('b1', HIDDEN)
    tracker.write('W2', HIDDEN)
    tracker.write('b2', 1)

    if grokfast:
        tracker.write('ema_W1', HIDDEN * n_bits)
        tracker.write('ema_b1', HIDDEN)
        tracker.write('ema_W2', HIDDEN)
        tracker.write('ema_b2', 1)
        ema_W1 = np.zeros_like(W1)
        ema_b1 = np.zeros_like(b1)
        ema_W2 = np.zeros_like(W2)
        ema_b2 = np.zeros_like(b2)

    xb = x_all[:BATCH_SIZE]
    yb = y_all[:BATCH_SIZE]
    bs = BATCH_SIZE

    tracker.write('xb', BATCH_SIZE * n_bits)
    tracker.write('yb', BATCH_SIZE)

    # --- Forward ---
    # h_pre = xb @ W1.T + b1
    tracker.read('xb', BATCH_SIZE * n_bits)
    tracker.read('W1', HIDDEN * n_bits)
    tracker.read('b1', HIDDEN)
    h_pre = xb @ W1.T + b1
    tracker.write('h_pre', BATCH_SIZE * HIDDEN)

    # h = relu(h_pre)
    tracker.read('h_pre', BATCH_SIZE * HIDDEN)
    h = np.maximum(h_pre, 0)
    tracker.write('h', BATCH_SIZE * HIDDEN)

    # out = h @ W2.T + b2
    tracker.read('h', BATCH_SIZE * HIDDEN)
    tracker.read('W2', HIDDEN)
    tracker.read('b2', 1)
    out = (h @ W2.T + b2).ravel()
    tracker.write('out', BATCH_SIZE)

    # margin = out * yb
    tracker.read('out', BATCH_SIZE)
    tracker.read('yb', BATCH_SIZE)
    margin = out * yb
    mask = margin < 1.0
    ms = int(np.sum(mask))

    if ms == 0:
        return 0.0, tracker.summary()

    xm = xb[mask]
    ym = yb[mask]
    hm = h[mask]
    h_pre_m = h_pre[mask]

    # --- Backward ---
    # dout = -ym
    tracker.read('yb', ms)  # read masked subset
    dout = -ym
    tracker.write('dout', ms)

    # gW2 = (dout[:, None] * hm).sum() / bs
    tracker.read('dout', ms)
    tracker.read('h', ms * HIDDEN)  # read masked h
    gW2 = (dout[:, None] * hm).sum(axis=0, keepdims=True) / bs
    tracker.write('gW2', HIDDEN)

    # gb2 = dout.sum() / bs
    tracker.read('dout', ms)
    gb2_val = dout.sum() / bs
    tracker.write('gb2', 1)

    # dh = dout[:, None] * W2
    tracker.read('dout', ms)
    tracker.read('W2', HIDDEN)
    dh = dout[:, None] * W2
    tracker.write('dh', ms * HIDDEN)

    # dh_pre = dh * (h_pre_m > 0)
    tracker.read('dh', ms * HIDDEN)
    tracker.read('h_pre', ms * HIDDEN)  # read masked h_pre
    dh_pre = dh * (h_pre_m > 0)
    tracker.write('dh_pre', ms * HIDDEN)

    # gW1 = dh_pre.T @ xm / bs
    tracker.read('dh_pre', ms * HIDDEN)
    tracker.read('xb', ms * n_bits)  # read masked x
    gW1 = (dh_pre.T @ xm) / bs
    tracker.write('gW1', HIDDEN * n_bits)

    # gb1 = dh_pre.sum(axis=0) / bs
    tracker.read('dh_pre', ms * HIDDEN)
    gb1_val = dh_pre.sum(axis=0) / bs
    tracker.write('gb1', HIDDEN)

    # --- GrokFast EMA filter ---
    if grokfast:
        tracker.read('ema_W1', HIDDEN * n_bits)
        tracker.read('gW1', HIDDEN * n_bits)
        ema_W1 = alpha * ema_W1 + (1 - alpha) * gW1
        tracker.write('ema_W1', HIDDEN * n_bits)

        tracker.read('ema_b1', HIDDEN)
        tracker.read('gb1', HIDDEN)
        ema_b1 = alpha * ema_b1 + (1 - alpha) * gb1_val
        tracker.write('ema_b1', HIDDEN)

        tracker.read('ema_W2', HIDDEN)
        tracker.read('gW2', HIDDEN)
        ema_W2 = alpha * ema_W2 + (1 - alpha) * gW2
        tracker.write('ema_W2', HIDDEN)

        tracker.read('ema_b2', 1)
        tracker.read('gb2', 1)
        ema_b2 = alpha * ema_b2 + (1 - alpha) * gb2_val
        tracker.write('ema_b2', 1)

        # Filtered gradients
        tracker.read('gW1', HIDDEN * n_bits)
        tracker.read('ema_W1', HIDDEN * n_bits)
        gW1 = gW1 + lam * ema_W1
        tracker.write('gW1', HIDDEN * n_bits)

        tracker.read('gb1', HIDDEN)
        tracker.read('ema_b1', HIDDEN)
        gb1_val = gb1_val + lam * ema_b1
        tracker.write('gb1', HIDDEN)

        tracker.read('gW2', HIDDEN)
        tracker.read('ema_W2', HIDDEN)
        gW2 = gW2 + lam * ema_W2
        tracker.write('gW2', HIDDEN)

        tracker.read('gb2', 1)
        tracker.read('ema_b2', 1)
        gb2_val = gb2_val + lam * ema_b2
        tracker.write('gb2', 1)

    # --- Weight updates ---
    # W1 -= lr * (gW1 + wd * W1)
    tracker.read('gW1', HIDDEN * n_bits)
    tracker.read('W1', HIDDEN * n_bits)
    W1 -= LR * (gW1 + WD * W1)
    tracker.write('W1', HIDDEN * n_bits)

    # b1 -= lr * (gb1 + wd * b1)
    tracker.read('gb1', HIDDEN)
    tracker.read('b1', HIDDEN)
    b1 -= LR * (gb1_val + WD * b1)
    tracker.write('b1', HIDDEN)

    # W2 -= lr * (gW2 + wd * W2)
    tracker.read('gW2', HIDDEN)
    tracker.read('W2', HIDDEN)
    W2 -= LR * (gW2 + WD * W2)
    tracker.write('W2', HIDDEN)

    # b2 -= lr * (gb2 + wd * b2)
    tracker.read('gb2', 1)
    tracker.read('b2', 1)
    b2 -= LR * (gb2_val + WD * b2)
    tracker.write('b2', 1)

    return tracker.summary()['dmd'], tracker.summary()


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  EXPERIMENT: GrokFast + Curriculum DMD Measurement")
    print("  Buffer-level LRU stack tracking (true LRU order, weighted by size)")
    print("=" * 78)

    rng = np.random.RandomState(SEED)
    secret_k3 = sorted(rng.choice(10, 3, replace=False).tolist())
    secret_k5 = sorted(rng.choice(10, 5, replace=False).tolist())

    steps_per_epoch = N_TRAIN // BATCH_SIZE  # 62

    # Convergence data from previous experiments
    regimes = [
        {
            "label": "n=20/k=5", "n": 20, "k": 5, "secret": secret_k5,
            "methods": {
                "sgd":          {"epochs": 70,   "gf": False, "curriculum": False},
                "grokfast":     {"epochs": 25,   "gf": True,  "curriculum": False},
                "curriculum":   {"epochs": 25,   "gf": False, "curriculum": True},
                "gf_curriculum":{"epochs": 12,   "gf": True,  "curriculum": True},
            },
        },
        {
            "label": "n=50/k=3", "n": 50, "k": 3, "secret": secret_k3,
            "methods": {
                "sgd":          {"epochs": 58,   "gf": False, "curriculum": False},
                "grokfast":     {"epochs": 148,  "gf": True,  "curriculum": False},
                "curriculum":   {"epochs": 10,   "gf": False, "curriculum": True},
                "gf_curriculum":{"epochs": 7,    "gf": True,  "curriculum": True},
            },
        },
        {
            "label": "n=50/k=5", "n": 50, "k": 5, "secret": secret_k5,
            "methods": {
                "sgd":          {"epochs": 1000, "gf": False, "curriculum": False},
                "grokfast":     {"epochs": 1000, "gf": True,  "curriculum": False},
                "curriculum":   {"epochs": 34,   "gf": False, "curriculum": True},
                "gf_curriculum":{"epochs": 14,   "gf": True,  "curriculum": True},
            },
        },
    ]

    all_results = {}

    for regime in regimes:
        n, k, secret = regime["n"], regime["k"], regime["secret"]
        print(f"\n{'─'*78}")
        print(f"  {regime['label']}  secret={secret}")
        print(f"{'─'*78}")

        regime_results = {}

        for method_name, cfg in regime["methods"].items():
            # For curriculum methods, measure at initial n=10
            if cfg["curriculum"]:
                dmd_small, _ = tracked_sgd_step(10, k, secret, SEED, cfg["gf"], GF_ALPHA, GF_LAM)
                dmd_large, summary = tracked_sgd_step(n, k, secret, SEED, cfg["gf"], GF_ALPHA, GF_LAM)
                # Most epochs at small n, ~2 at large n per expansion phase
                small_epochs = max(1, cfg["epochs"] - 2)
                large_epochs = 2
                total_dmd = (dmd_small * small_epochs + dmd_large * large_epochs) * steps_per_epoch
                regime_results[method_name] = {
                    "dmd_per_step_n10": round(dmd_small, 1),
                    "dmd_per_step_target": round(dmd_large, 1),
                    "epochs": cfg["epochs"],
                    "total_dmd": round(total_dmd, 0),
                }
                print(f"  {method_name:<20} DMD/step(n=10)={dmd_small:>10,.1f}  "
                      f"DMD/step(n={n})={dmd_large:>10,.1f}  "
                      f"epochs={cfg['epochs']:>5}  total_DMD={total_dmd:>14,.0f}")
            else:
                dmd_step, summary = tracked_sgd_step(n, k, secret, SEED, cfg["gf"], GF_ALPHA, GF_LAM)
                total_steps = cfg["epochs"] * steps_per_epoch
                total_dmd = dmd_step * total_steps
                regime_results[method_name] = {
                    "dmd_per_step": round(dmd_step, 1),
                    "epochs": cfg["epochs"],
                    "total_steps": total_steps,
                    "total_dmd": round(total_dmd, 0),
                }
                print(f"  {method_name:<20} DMD/step={dmd_step:>10,.1f}  "
                      f"epochs={cfg['epochs']:>5}  total_DMD={total_dmd:>14,.0f}")

        all_results[regime["label"]] = regime_results

    # Summary
    print(f"\n{'='*78}")
    print(f"  SUMMARY: Total DMD (lower = more energy efficient)")
    print(f"{'='*78}")
    print(f"  {'Regime':<12} {'Method':<20} {'Epochs':>8} {'Total DMD':>14} {'vs SGD':>10}")
    print(f"  {'─'*12} {'─'*20} {'─'*8} {'─'*14} {'─'*10}")

    for regime_label, methods in all_results.items():
        sgd_dmd = methods.get("sgd", {}).get("total_dmd", 1)
        for method_name, r in methods.items():
            ratio = f"{r['total_dmd']/sgd_dmd:.2f}x" if sgd_dmd > 0 else "N/A"
            print(f"  {regime_label:<12} {method_name:<20} {r['epochs']:>8} "
                  f"{r['total_dmd']:>14,.0f} {ratio:>10}")
        print()

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {RESULTS_DIR / 'results.json'}")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()

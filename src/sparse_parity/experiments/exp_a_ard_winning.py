"""
Experiment A: Measure ARD on the winning 20-bit sparse parity config.

We solved 20-bit sparse parity (k=3) with LR=0.1, batch_size=32, n_train=500.
Now we measure Average Reuse Distance (ARD) for each of the 3 training variants
(standard backprop, fused, per-layer) to see which is most energy-efficient.

Approach:
  1. For each variant, train with single-sample SGD until >90% test accuracy
     (or max 200 epochs).
  2. Instrument the FIRST step of the next epoch with MemTracker.
  3. Record and compare weighted ARD across variants.

NOTE: Hidden reduced from 1000 to 500 for runtime (<5 min total).
"""

import sys
import os
import time
import json
import copy
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params, forward, forward_batch
from sparse_parity.metrics import hinge_loss, accuracy, save_json, timestamp
from sparse_parity.tracker import MemTracker
from sparse_parity.train import backward_and_update
from sparse_parity.train_fused import backward_and_update_fused
from sparse_parity.train_perlayer import train_step_perlayer, forward_batch_perlayer


# ---------------------------------------------------------------------------
# Configuration: winning config (hidden=500 for speed)
# ---------------------------------------------------------------------------

HIDDEN = 500
CONFIG = Config(
    n_bits=20,
    k_sparse=3,
    n_train=500,
    n_test=200,
    hidden=HIDDEN,
    lr=0.1,
    wd=0.01,
    max_epochs=200,
    seed=42,
)


# ---------------------------------------------------------------------------
# Phase 1: Train until convergence (no tracking)
# ---------------------------------------------------------------------------

def train_until_converged(variant, config):
    """
    Train with single-sample SGD (no batching) until >90% test accuracy.
    Returns (W1, b1, W2, b2, epochs_to_converge, best_test_acc, elapsed).

    variant: 'standard' | 'fused' | 'perlayer'
    """
    x_train, y_train, x_test, y_test, secret = generate(config)
    W1, b1, W2, b2 = init_params(config)

    best_test_acc = 0.0
    converge_epoch = None
    start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        for i in range(len(x_train)):
            if variant == 'standard':
                out, h_pre, h = forward(x_train[i], W1, b1, W2, b2)
                backward_and_update(x_train[i], y_train[i], out, h_pre, h,
                                    W1, b1, W2, b2, config)
            elif variant == 'fused':
                out, h_pre, h = forward(x_train[i], W1, b1, W2, b2)
                backward_and_update_fused(x_train[i], y_train[i], out, h_pre, h,
                                          W1, b1, W2, b2, config)
            elif variant == 'perlayer':
                train_step_perlayer(x_train[i], y_train[i], W1, b1, W2, b2, config)
            else:
                raise ValueError(f"Unknown variant: {variant}")

        # Evaluate
        if variant == 'perlayer':
            te_outs = forward_batch_perlayer(x_test, W1, b1, W2, b2, config)
            tr_outs = forward_batch_perlayer(x_train, W1, b1, W2, b2, config)
        else:
            te_outs = forward_batch(x_test, W1, b1, W2, b2)
            tr_outs = forward_batch(x_train, W1, b1, W2, b2)

        te_acc = accuracy(te_outs, y_test)
        tr_acc = accuracy(tr_outs, y_train)

        if te_acc > best_test_acc:
            best_test_acc = te_acc

        if epoch % 20 == 0 or epoch == 1 or te_acc > 0.9:
            elapsed = time.time() - start
            print(f"    Epoch {epoch:4d} | train_acc={tr_acc:.3f} test_acc={te_acc:.3f} | {elapsed:.1f}s")

        if best_test_acc >= 0.90:
            if converge_epoch is None:
                converge_epoch = epoch
            # Keep going one more epoch to ensure stability, but break
            break

    elapsed = time.time() - start
    if converge_epoch is None:
        converge_epoch = config.max_epochs

    return W1, b1, W2, b2, converge_epoch, best_test_acc, elapsed


# ---------------------------------------------------------------------------
# Phase 2: Instrument one step with MemTracker
# ---------------------------------------------------------------------------

def instrument_one_step(variant, W1, b1, W2, b2, config):
    """
    Run ONE single-sample training step with MemTracker instrumentation.
    Finds a sample where margin < 1 (gradient will fire) to get a full
    forward+backward trace. If no such sample exists, uses sample 0.
    Returns tracker summary dict.
    """
    x_train, y_train, _, _, _ = generate(config)

    # Find a sample where margin < 1 so backward actually executes
    chosen_idx = 0
    for i in range(len(x_train)):
        x = x_train[i]
        hidden = len(W1)
        n_bits = len(x)
        h_pre = [sum(W1[j][ii] * x[ii] for ii in range(n_bits)) + b1[j] for j in range(hidden)]
        h = [max(0.0, v) for v in h_pre]
        out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]
        margin = out * y_train[i]
        if margin < 1.0:
            chosen_idx = i
            print(f"    Found sample {i} with margin={margin:.4f} < 1 (gradient will fire)")
            break
    else:
        print(f"    WARNING: All samples have margin >= 1. Using sample 0 (forward-only trace).")

    x = x_train[chosen_idx]
    y = y_train[chosen_idx]

    tracker = MemTracker()

    # Write initial state of all buffers
    tracker.write('W1', config.hidden * config.n_bits)
    tracker.write('b1', config.hidden)
    tracker.write('W2', config.hidden)
    tracker.write('b2', 1)
    tracker.write('x', config.n_bits)
    tracker.write('y', 1)

    if variant == 'standard':
        out, h_pre, h = forward(x, W1, b1, W2, b2, tracker=tracker)
        backward_and_update(x, y, out, h_pre, h, W1, b1, W2, b2, config, tracker=tracker)
    elif variant == 'fused':
        out, h_pre, h = forward(x, W1, b1, W2, b2, tracker=tracker)
        backward_and_update_fused(x, y, out, h_pre, h, W1, b1, W2, b2, config, tracker=tracker)
    elif variant == 'perlayer':
        train_step_perlayer(x, y, W1, b1, W2, b2, config, tracker=tracker)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    tracker.report()
    return tracker.to_json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  EXPERIMENT A: Measure ARD on Winning 20-bit Sparse Parity Config")
    print("=" * 70)
    print(f"  n_bits={CONFIG.n_bits}, k_sparse={CONFIG.k_sparse}, hidden={CONFIG.hidden}")
    print(f"  n_train={CONFIG.n_train}, n_test={CONFIG.n_test}")
    print(f"  lr={CONFIG.lr}, wd={CONFIG.wd}")
    print(f"  max_epochs={CONFIG.max_epochs}, seed={CONFIG.seed}")
    print("=" * 70)
    print()

    variants = ['standard', 'fused', 'perlayer']
    results = {}

    for variant in variants:
        print(f"\n{'─' * 70}")
        print(f"  Training variant: {variant}")
        print(f"{'─' * 70}")

        # Train to convergence
        W1, b1, W2, b2, epochs, best_acc, elapsed = train_until_converged(variant, CONFIG)
        print(f"\n  Converged: epoch={epochs}, best_test_acc={best_acc:.3f}, time={elapsed:.1f}s")

        # Deep copy weights so instrumentation step doesn't corrupt state
        W1_copy = [row[:] for row in W1]
        b1_copy = b1[:]
        W2_copy = [row[:] for row in W2]
        b2_copy = b2[:]

        # Instrument one step
        print(f"\n  Instrumenting one step with MemTracker...")
        tracker_data = instrument_one_step(variant, W1_copy, b1_copy, W2_copy, b2_copy, CONFIG)

        results[variant] = {
            'epochs_to_converge': epochs,
            'best_test_acc': best_acc,
            'training_time_s': elapsed,
            'tracker': tracker_data,
        }

    # ---------------------------------------------------------------------------
    # Comparison table
    # ---------------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  ARD COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Variant':<16} {'Epochs':>7} {'Test Acc':>9} {'Weighted ARD':>14} {'Reads':>7} {'Writes':>7} {'Total Floats':>14}")
    print(f"  {'─'*16} {'─'*7} {'─'*9} {'─'*14} {'─'*7} {'─'*7} {'─'*14}")

    for variant in variants:
        r = results[variant]
        t = r['tracker']
        print(f"  {variant:<16} {r['epochs_to_converge']:>7} {r['best_test_acc']:>9.3f} "
              f"{t['weighted_ard']:>14,.0f} {t['reads']:>7} {t['writes']:>7} "
              f"{t['total_floats_accessed']:>14,}")

    # Compute improvement ratios
    std_ard = results['standard']['tracker']['weighted_ard']
    if std_ard > 0:
        print(f"\n  ARD improvement vs standard backprop:")
        for variant in ['fused', 'perlayer']:
            v_ard = results[variant]['tracker']['weighted_ard']
            if v_ard > 0:
                ratio = std_ard / v_ard
                pct = (1 - v_ard / std_ard) * 100
                print(f"    {variant:<16}: {ratio:.2f}x better ({pct:.1f}% reduction)")
            else:
                print(f"    {variant:<16}: N/A (ARD=0, likely no-gradient step)")

    print("=" * 70)

    # Save results
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_a_ard_winning'
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment': 'exp_a_ard_winning',
        'description': 'Measure ARD on winning 20-bit sparse parity config across 3 training variants',
        'config': {
            'n_bits': CONFIG.n_bits,
            'k_sparse': CONFIG.k_sparse,
            'n_train': CONFIG.n_train,
            'n_test': CONFIG.n_test,
            'hidden': CONFIG.hidden,
            'lr': CONFIG.lr,
            'wd': CONFIG.wd,
            'max_epochs': CONFIG.max_epochs,
            'seed': CONFIG.seed,
        },
        'variants': results,
    }

    results_path = results_dir / 'results.json'
    save_json(output, results_path)
    print(f"\n  Results saved to: {results_path}")

    return output


if __name__ == '__main__':
    main()

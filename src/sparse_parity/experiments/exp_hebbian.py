#!/usr/bin/env python3
"""
Experiment: Hebbian + Anti-Hebbian Learning for Sparse Parity

Hypothesis: Purely local Hebbian learning rules (no backprop, no global error
signal) can extract useful features from sparse parity data. Weight updates
depend only on pre- and post-synaptic activity.

Risk: Standard Hebbian learning finds linear correlations, and parity bits
have zero linear correlation with the label. Need nonlinear Hebbian variants
or multi-layer approach.

Variants tested:
  1. Simple Hebb with decay: dW1 = lr * (x * h^T - alpha * W1)
  2. Oja's rule (normalized Hebb): dW1 = lr * h * (x - W1^T * h)
  3. BCM rule (sliding threshold): dW1 = lr * x * h * (h - theta)

Layer 2 is always supervised: dW2 = lr * (y - y_hat) * h

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_hebbian.py
"""

import sys
import time
import json
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparse_parity.tracker import MemTracker


# =============================================================================
# DATA GENERATION (numpy-based for speed)
# =============================================================================

def generate_data(n_bits, k_sparse, n_train, n_test, seed=42):
    """Generate sparse parity train/test data. Returns x_train, y_train, x_test, y_test, secret."""
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())

    x_train = rng.choice([-1.0, 1.0], size=(n_train, n_bits))
    y_train = np.prod(x_train[:, secret], axis=1)

    x_test = rng.choice([-1.0, 1.0], size=(n_test, n_bits))
    y_test = np.prod(x_test[:, secret], axis=1)

    return x_train, y_train, x_test, y_test, secret


# =============================================================================
# HEBBIAN LAYER 1 VARIANTS (unsupervised, local updates)
# =============================================================================

def hebbian_forward(x, W1, b1):
    """Forward pass through layer 1: h = tanh(W1 @ x + b1).

    Using tanh (not ReLU) because Hebbian rules work better with
    bipolar activations that can be anti-correlated.
    """
    h_pre = W1 @ x + b1
    h = np.tanh(h_pre)
    return h_pre, h


def simple_hebb_update(x, h, W1, b1, lr, alpha=0.01):
    """Simple Hebbian with weight decay (anti-Hebbian regularization).

    dW1 = lr * (outer(h, x) - alpha * W1)
    db1 = lr * (h - alpha * b1)

    The decay term alpha * W1 prevents unbounded growth and provides
    anti-Hebbian regularization for uncorrelated pairs.
    """
    W1 += lr * (np.outer(h, x) - alpha * W1)
    b1 += lr * (h - alpha * b1)


def oja_update(x, h, W1, b1, lr):
    """Oja's rule: normalized Hebbian learning.

    dW_j = lr * h_j * (x - h_j * W_j)

    This extracts principal components. Each neuron converges to a
    unit-norm weight vector pointing along a principal direction.
    """
    hidden = len(h)
    for j in range(hidden):
        W1[j] += lr * h[j] * (x - h[j] * W1[j])
        b1[j] += lr * h[j] * (1.0 - h[j] * b1[j])


def bcm_update(x, h, W1, b1, lr, theta, tau_theta=100.0):
    """BCM (Bienenstock-Cooper-Munro) rule with sliding threshold.

    dW_j = lr * x * h_j * (h_j - theta_j)
    theta_j is a sliding average of h_j^2

    When h > theta: potentiation (Hebbian)
    When h < theta: depression (anti-Hebbian)
    This creates selectivity, not just correlation detection.
    """
    hidden = len(h)
    for j in range(hidden):
        phi = h[j] * (h[j] - theta[j])
        W1[j] += lr * phi * x
        b1[j] += lr * phi
        # Update sliding threshold
        theta[j] += (h[j] ** 2 - theta[j]) / tau_theta


# =============================================================================
# LAYER 2: SUPERVISED PERCEPTRON (local with label signal)
# =============================================================================

def perceptron_forward(h, W2, b2):
    """Output layer: y_hat = W2 @ h + b2 (single scalar output)."""
    return float(W2 @ h + b2)


def perceptron_update(h, y, y_hat, W2, b2, lr):
    """Perceptron update rule: dW2 = lr * (y - y_hat) * h.

    This is local in the sense that the error signal (y - y_hat) is
    computed at the output and only affects the output weights.
    """
    err = y - y_hat
    W2[0] += lr * err * h
    b2[0] += lr * err


# =============================================================================
# MEMORY TRACKER INSTRUMENTATION
# =============================================================================

def instrument_one_step(x_sample, y_sample, W1, b1, W2, b2, rule_name):
    """Run one training step with MemTracker to measure ARD."""
    n_bits = len(x_sample)
    hidden = W1.shape[0]

    tracker = MemTracker()

    # Initial buffer writes (all params in memory)
    tracker.write('W1', hidden * n_bits)
    tracker.write('b1', hidden)
    tracker.write('W2', hidden)
    tracker.write('b2', 1)
    tracker.write('x', n_bits)
    tracker.write('y', 1)

    # Forward layer 1
    tracker.read('x', n_bits)
    tracker.read('W1', hidden * n_bits)
    tracker.read('b1', hidden)
    h_pre, h = hebbian_forward(x_sample, W1, b1)
    tracker.write('h', hidden)

    # Hebbian update layer 1 (local: only reads x, h, W1)
    tracker.read('h', hidden)
    tracker.read('x', n_bits)
    tracker.read('W1', hidden * n_bits)
    tracker.read('b1', hidden)
    # (update happens here -- W1, b1 modified in place)
    tracker.write('W1', hidden * n_bits)
    tracker.write('b1', hidden)

    # Forward layer 2
    tracker.read('h', hidden)
    tracker.read('W2', hidden)
    tracker.read('b2', 1)
    y_hat = perceptron_forward(h, W2, b2)
    tracker.write('y_hat', 1)

    # Perceptron update layer 2 (local: only reads h, y, y_hat)
    tracker.read('y', 1)
    tracker.read('y_hat', 1)
    tracker.read('h', hidden)
    tracker.read('W2', hidden)
    # (update happens here)
    tracker.write('W2', hidden)
    tracker.read('b2', 1)
    tracker.write('b2', 1)

    return tracker


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_hebbian(x_train, y_train, x_test, y_test, n_bits, hidden, lr,
                  max_epochs, rule='simple_hebb', alpha=0.01, seed=42):
    """Train a 2-layer Hebbian network.

    Layer 1: unsupervised Hebbian (one of three variants)
    Layer 2: supervised perceptron

    Returns dict with accuracy history and final metrics.
    """
    rng = np.random.RandomState(seed + 1)

    # Initialize weights
    std1 = math.sqrt(2.0 / n_bits)
    W1 = rng.randn(hidden, n_bits) * std1
    b1 = np.zeros(hidden)

    std2 = math.sqrt(2.0 / hidden)
    W2 = rng.randn(1, hidden) * std2
    b2 = np.zeros(1)

    # BCM sliding threshold
    theta = np.ones(hidden) * 0.5

    n_train = len(x_train)
    n_test = len(x_test)
    train_accs = []
    test_accs = []
    best_test_acc = 0.0
    converge_epoch = None

    start = time.time()

    for epoch in range(1, max_epochs + 1):
        # Shuffle training data
        perm = rng.permutation(n_train)

        for idx in perm:
            x_i = x_train[idx]
            y_i = y_train[idx]

            # Forward layer 1
            h_pre, h = hebbian_forward(x_i, W1, b1)

            # Hebbian update layer 1 (unsupervised)
            if rule == 'simple_hebb':
                simple_hebb_update(x_i, h, W1, b1, lr, alpha=alpha)
            elif rule == 'oja':
                oja_update(x_i, h, W1, b1, lr)
            elif rule == 'bcm':
                bcm_update(x_i, h, W1, b1, lr, theta)
            else:
                raise ValueError(f"Unknown rule: {rule}")

            # Forward layer 2
            y_hat = perceptron_forward(h, W2, b2)

            # Perceptron update layer 2 (supervised)
            perceptron_update(h, y_i, y_hat, W2, b2, lr)

        # Evaluate
        train_preds = []
        for i in range(n_train):
            _, h = hebbian_forward(x_train[i], W1, b1)
            y_hat = perceptron_forward(h, W2, b2)
            train_preds.append(1.0 if y_hat >= 0 else -1.0)
        train_acc = np.mean(np.array(train_preds) == y_train)

        test_preds = []
        for i in range(n_test):
            _, h = hebbian_forward(x_test[i], W1, b1)
            y_hat = perceptron_forward(h, W2, b2)
            test_preds.append(1.0 if y_hat >= 0 else -1.0)
        test_acc = np.mean(np.array(test_preds) == y_test)

        train_accs.append(float(train_acc))
        test_accs.append(float(test_acc))

        if test_acc > best_test_acc:
            best_test_acc = float(test_acc)

        if best_test_acc >= 0.9 and converge_epoch is None:
            converge_epoch = epoch

        if best_test_acc >= 1.0:
            break

        # Timeout guard: 30 seconds per config
        if time.time() - start > 30:
            break

    elapsed = time.time() - start

    # Instrument one step for ARD
    tracker = instrument_one_step(x_train[0], y_train[0], W1, b1, W2, b2, rule)

    return {
        'rule': rule,
        'hidden': hidden,
        'lr': lr,
        'alpha': alpha,
        'max_epochs': max_epochs,
        'epochs_run': epoch,
        'converge_epoch': converge_epoch,
        'best_test_acc': best_test_acc,
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1],
        'train_accs': train_accs,
        'test_accs': test_accs,
        'elapsed_s': round(elapsed, 4),
        'ard': tracker.to_json(),
    }


# =============================================================================
# RUN ONE CONFIG ACROSS SEEDS AND RULES
# =============================================================================

def run_config(n_bits, k_sparse, hidden, lr, max_epochs, n_train, n_test,
               seeds, alpha=0.01, verbose=True):
    """Run all three Hebbian rules on one config with multiple seeds."""
    rules = ['simple_hebb', 'oja', 'bcm']
    results = {}

    for rule in rules:
        rule_results = []
        for seed in seeds:
            x_train, y_train, x_test, y_test, secret = generate_data(
                n_bits, k_sparse, n_train, n_test, seed=seed
            )

            if verbose:
                print(f"    [{rule}] seed={seed}, secret={secret}...", end=" ", flush=True)

            res = train_hebbian(
                x_train, y_train, x_test, y_test,
                n_bits=n_bits, hidden=hidden, lr=lr,
                max_epochs=max_epochs, rule=rule, alpha=alpha, seed=seed
            )

            if verbose:
                status = f"test_acc={res['best_test_acc']:.3f}"
                if res['converge_epoch']:
                    status += f" (converged epoch {res['converge_epoch']})"
                print(status)

            rule_results.append({
                'seed': seed,
                'secret': secret,
                'best_test_acc': res['best_test_acc'],
                'final_train_acc': res['final_train_acc'],
                'final_test_acc': res['final_test_acc'],
                'converge_epoch': res['converge_epoch'],
                'epochs_run': res['epochs_run'],
                'elapsed_s': res['elapsed_s'],
                'ard': res['ard'],
            })

        results[rule] = rule_results

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: Hebbian + Anti-Hebbian Learning for Sparse Parity")
    print("  Approach #7: Purely local learning rules, no backprop")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # -------------------------------------------------------------------
    # Config 1: n=20, k=3, hidden=1000, lr=0.1
    # -------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  CONFIG 1: n=20, k=3, hidden=1000, lr=0.1")
    print(f"{'=' * 70}")

    all_results['n20_k3_h1000'] = run_config(
        n_bits=20, k_sparse=3, hidden=1000, lr=0.1,
        max_epochs=200, n_train=500, n_test=200,
        seeds=seeds, alpha=0.01
    )

    # -------------------------------------------------------------------
    # Config 2: n=20, k=3, hidden=200, lr=0.1
    # -------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  CONFIG 2: n=20, k=3, hidden=200, lr=0.1")
    print(f"{'=' * 70}")

    all_results['n20_k3_h200'] = run_config(
        n_bits=20, k_sparse=3, hidden=200, lr=0.1,
        max_epochs=200, n_train=500, n_test=200,
        seeds=seeds, alpha=0.01
    )

    # -------------------------------------------------------------------
    # Config 3: Smaller lr for stability (some rules blow up at lr=0.1)
    # -------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  CONFIG 3: n=20, k=3, hidden=1000, lr=0.01 (stability test)")
    print(f"{'=' * 70}")

    all_results['n20_k3_h1000_lowlr'] = run_config(
        n_bits=20, k_sparse=3, hidden=1000, lr=0.01,
        max_epochs=200, n_train=500, n_test=200,
        seeds=seeds, alpha=0.001
    )

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print(f"\n\n{'=' * 90}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 90}")
    header = (f"  {'Config':<25} | {'Rule':<12} | {'Best Acc':>9} | "
              f"{'Avg Acc':>9} | {'Converged':>9} | {'Avg ARD':>12}")
    print(header)
    print("  " + "-" * 88)

    for config_key, config_results in all_results.items():
        for rule, rule_results in config_results.items():
            accs = [r['best_test_acc'] for r in rule_results]
            best_acc = max(accs)
            avg_acc = np.mean(accs)
            n_converged = sum(1 for r in rule_results if r['converge_epoch'] is not None)
            avg_ard = np.mean([r['ard']['weighted_ard'] for r in rule_results])

            print(f"  {config_key:<25} | {rule:<12} | {best_acc:>8.3f} | "
                  f"{avg_acc:>8.3f} | {n_converged}/{len(rule_results):>7} | "
                  f"{avg_ard:>11,.0f}")

    # -------------------------------------------------------------------
    # Analysis of why Hebbian fails on parity
    # -------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  ANALYSIS: Why Hebbian learning fails on sparse parity")
    print(f"{'=' * 70}")

    # Check if any rule beat chance (50%)
    any_above_chance = False
    for config_key, config_results in all_results.items():
        for rule, rule_results in config_results.items():
            best = max(r['best_test_acc'] for r in rule_results)
            if best > 0.55:
                any_above_chance = True
                print(f"  {config_key}/{rule}: best={best:.3f} (above chance)")

    if not any_above_chance:
        print("  No rule exceeded 55% accuracy (chance = 50%).")
        print("  This confirms the theoretical prediction:")
        print("    - Parity bits have ZERO linear correlation with the label")
        print("    - Hebbian rules (all variants) learn linear correlations")
        print("    - Even Oja's PCA and BCM selectivity cannot find XOR-like structure")
        print("    - The supervised perceptron on layer 2 also fails because")
        print("      the Hebbian features don't capture parity information")

    # Compare ARD with backprop (from known results)
    print(f"\n{'=' * 70}")
    print("  ARD COMPARISON (Hebbian vs Backprop)")
    print(f"{'=' * 70}")
    print("  Hebbian ARD is expected to be LOW because:")
    print("    - No backward pass (no gradient chain)")
    print("    - Each layer updated independently")
    print("    - Only local reads/writes per layer")

    # Pick first config for comparison
    first_config = list(all_results.values())[0]
    for rule, results_list in first_config.items():
        ard = results_list[0]['ard']['weighted_ard']
        print(f"    {rule}: ARD = {ard:,.0f}")

    print(f"{'=' * 70}")

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_hebbian'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prepare JSON-serializable results (strip accuracy histories for brevity)
    save_results = {}
    for config_key, config_results in all_results.items():
        save_results[config_key] = {}
        for rule, rule_results in config_results.items():
            save_results[config_key][rule] = []
            for r in rule_results:
                save_entry = {k: v for k, v in r.items()}
                # Keep only last 10 accuracy values to save space
                save_entry.pop('train_accs', None)
                save_entry.pop('test_accs', None)
                save_results[config_key][rule].append(save_entry)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_hebbian',
            'description': 'Hebbian + Anti-Hebbian learning for sparse parity',
            'hypothesis': 'Purely local Hebbian rules cannot solve parity (zero linear correlation), but will have low ARD',
            'approach': 'Hebbian/anti-Hebbian, no backprop, no global error signal',
            'rules_tested': ['simple_hebb', 'oja', 'bcm'],
            'configs': save_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

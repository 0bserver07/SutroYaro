#!/usr/bin/env python3
"""
Experiment: Predictive Coding for Sparse Parity

Hypothesis: Predictive coding's local learning rule (each layer only talks to
its neighbors) yields smaller ARD than backprop, because the working set per
layer update is just its own weights + activations + one neighbor's prediction
error. Under certain conditions, predictive coding approximates backprop's
weight updates (Millidge et al. 2021), so accuracy should be comparable.

Architecture (2-layer, matching baseline):
  Layer 0: input (n_bits)
  Layer 1: hidden (ReLU)
  Layer 2: output (1, linear for regression toward {-1,+1})

Predictive coding loop per sample:
  1. Clamp x at layer 0 and y at output layer 2
  2. Initialize value nodes: forward pass to get initial guesses
  3. Inference phase (10-20 iterations): settle the network
     - Each layer computes prediction error = value - predicted_value
     - Update value nodes to minimize prediction errors (gradient on values)
  4. Weight update phase: use converged prediction errors for local dW
     - dW_l = lr * error_{l+1} * activation_l^T  (local Hebbian-like rule)

Reference: Rao & Ballard (1999), Millidge, Tschantz & Buckley (2021)

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_predictive_coding.py
"""

import sys
import math
import random
import time
import json
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params, forward
from sparse_parity.train import backward_and_update
from sparse_parity.metrics import accuracy, save_json
from sparse_parity.tracker import MemTracker


# ===========================================================================
# Predictive Coding Network (pure Python, no PyTorch/numpy)
# ===========================================================================

def pc_init_params(n_input, hidden, seed=42):
    """Initialize 2-layer predictive coding network: input -> hidden -> output(1).

    W1: hidden x n_input  (predicts layer 0 from layer 1 -- generative direction)
    W2: 1 x hidden        (predicts layer 1 from layer 2 -- generative direction)

    NOTE: In predictive coding, weights go in the GENERATIVE (top-down) direction.
    W1 predicts x from h, W2 predicts h from output.
    For the forward (recognition) pass we use the transpose.
    """
    rng = random.Random(seed + 200)

    std1 = math.sqrt(2.0 / n_input)
    W1 = [[rng.gauss(0, std1) for _ in range(n_input)] for _ in range(hidden)]
    b1 = [0.0] * hidden

    std2 = math.sqrt(2.0 / hidden)
    W2 = [rng.gauss(0, std2) for _ in range(hidden)]
    b2 = 0.0

    return W1, b1, W2, b2


def relu(x):
    return max(0.0, x)


def relu_deriv(x):
    return 1.0 if x > 0 else 0.0


def pc_forward_init(x, W1, b1, W2, b2):
    """Initialize value nodes via a single forward pass (using W^T as recognition weights).

    Returns initial value nodes: (mu0, mu1_pre, mu1, mu2)
    mu0 = x (clamped)
    mu1 = ReLU(W1 @ x + b1)  -- hidden activations
    mu2 = W2 @ mu1 + b2      -- output prediction
    """
    hidden = len(W1)
    n_input = len(x)

    # Layer 1: h_pre = W1 @ x + b1, h = ReLU(h_pre)
    mu1_pre = [sum(W1[j][i] * x[i] for i in range(n_input)) + b1[j]
               for j in range(hidden)]
    mu1 = [relu(v) for v in mu1_pre]

    # Layer 2: out = W2 . mu1 + b2
    mu2 = sum(W2[j] * mu1[j] for j in range(hidden)) + b2

    return mu1_pre, mu1, mu2


def pc_inference_step(x, y, mu1_pre, mu1, mu2, W1, b1, W2, b2, inf_lr):
    """One inference iteration: update value nodes mu1 to reduce prediction errors.

    Prediction errors:
      e0[i] = x[i] - pred0[i]       where pred0 = W1^T @ mu1 (layer 1 predicts layer 0)
      e1[j] = mu1[j] - pred1[j]     where pred1 = f(W2^T * mu2) (but we use linear here)
      e2    = y - mu2               (output clamped to target)

    Wait -- standard PC formulation:
      e_l = mu_l - g(theta_l * mu_{l+1})   (prediction error at layer l)
      g is the activation function

    So:
      e0 = x - g0(W1^T @ mu1)    -- but x is clamped, not updated
      e1 = mu1 - g1(W2^T * mu2)  -- prediction of hidden from output
      e2 = y - mu2               -- output error (clamped target)

    Value node update for mu1 (the free layer):
      d(mu1)/dt = -e1 + W1 @ e0 * g'(mu1_pre)
                = -(mu1 - W2^T * mu2) + W1 @ (x - W1^T @ mu1) * relu'(mu1_pre)

    For simplicity, update mu1 directly:
      mu1_new = mu1 + inf_lr * (-e1 + (d_prediction_error_0 / d_mu1))
    """
    hidden = len(W1)
    n_input = len(x)

    # Compute prediction of layer 0 from layer 1: pred0 = W1^T @ mu1
    # This is the generative prediction: mu1 -> x
    # Actually W1 is hidden x n_input, so W1[j][i] maps input i to hidden j
    # For generative direction: pred0[i] = sum_j W1[j][i] * mu1[j]
    pred0 = [sum(W1[j][i] * mu1[j] for j in range(hidden)) for i in range(n_input)]

    # Prediction error at layer 0: e0 = x - pred0
    e0 = [x[i] - pred0[i] for i in range(n_input)]

    # Prediction of layer 1 from layer 2: pred1[j] = W2[j] * mu2
    pred1 = [W2[j] * mu2 for j in range(hidden)]

    # Prediction error at layer 1: e1 = mu1 - pred1
    e1 = [mu1[j] - pred1[j] for j in range(hidden)]

    # Output error: e2 = y - mu2
    e2 = y - mu2

    # Update mu1 (the only free value node -- x is clamped, y is clamped)
    # Gradient from e0: W1 @ e0 (how changing mu1 affects prediction of x)
    # This is: sum_i W1[j][i] * e0[i] for each j
    grad_from_e0 = [sum(W1[j][i] * e0[i] for i in range(n_input)) for j in range(hidden)]

    # Gradient from e1: -e1 (direct error at this layer)
    # Gradient from e2 propagated through W2: W2[j] * e2
    grad_from_e2 = [W2[j] * e2 for j in range(hidden)]

    # Total gradient on mu1_pre (before activation)
    for j in range(hidden):
        d_mu1_pre = (-e1[j] + grad_from_e0[j] + grad_from_e2[j]) * relu_deriv(mu1_pre[j])
        mu1_pre[j] += inf_lr * d_mu1_pre
        mu1[j] = relu(mu1_pre[j])

    # Update mu2 (also a free node in some formulations, but we clamp to y)
    # In supervised PC, we clamp mu2 to y, so:
    mu2 = y  # Clamp output to target

    return mu1_pre, mu1, mu2, e0, e1, e2


def pc_weight_update(x, mu1, e0, e1, e2, W1, b1, W2, b2, lr, wd):
    """Update weights using converged prediction errors (local rule).

    Weight update rule (from Millidge et al. 2021):
      dW_l = lr * e_l @ mu_{l+1}^T   (for generative weights)

    For our network:
      dW1[j][i] = lr * e0[i] * mu1[j]    (W1 generates x from h)
      db1[j]    = lr * e1[j]              (bias update from layer 1 error)
      dW2[j]    = lr * e2 * mu1[j]        (W2 generates h from output)
      db2       = lr * e2
    """
    hidden = len(W1)
    n_input = len(x)

    # Update W1: dW1[j][i] = lr * e0[i] * mu1[j]
    for j in range(hidden):
        for i in range(n_input):
            W1[j][i] += lr * e0[i] * mu1[j] - wd * W1[j][i]

    # Update b1
    for j in range(hidden):
        b1[j] += lr * e1[j] - wd * b1[j]

    # Update W2: dW2[j] = lr * e2 * mu1[j]
    for j in range(hidden):
        W2[j] += lr * e2 * mu1[j] - wd * W2[j]

    # Update b2
    b2 += lr * e2 - wd * b2

    return W1, b1, W2, b2


def pc_train_step(x, y, W1, b1, W2, b2, lr, wd, n_inference_iters=15, inf_lr=0.1):
    """One predictive coding training step for a single sample.

    1. Forward init (get initial value node guesses)
    2. Inference phase: iterate to settle prediction errors
    3. Weight update: use converged errors for local weight updates
    """
    # 1. Initialize value nodes
    mu1_pre, mu1, mu2 = pc_forward_init(x, W1, b1, W2, b2)

    # 2. Inference phase: settle the network
    for t in range(n_inference_iters):
        mu1_pre, mu1, mu2, e0, e1, e2 = pc_inference_step(
            x, y, mu1_pre, mu1, mu2, W1, b1, W2, b2, inf_lr
        )

    # 3. Weight update using converged errors
    W1, b1, W2, b2 = pc_weight_update(x, mu1, e0, e1, e2, W1, b1, W2, b2, lr, wd)

    return W1, b1, W2, b2


def pc_predict(x, W1, b1, W2, b2):
    """Predict output for a single input (just a forward pass)."""
    _, mu1, mu2 = pc_forward_init(x, W1, b1, W2, b2)
    return mu2


def pc_accuracy(xs, ys, W1, b1, W2, b2):
    """Compute accuracy: sign(prediction) == y."""
    correct = 0
    for x, y in zip(xs, ys):
        pred = pc_predict(x, W1, b1, W2, b2)
        if (1.0 if pred >= 0 else -1.0) == y:
            correct += 1
    return correct / len(ys)


# ===========================================================================
# ARD instrumentation for Predictive Coding
# ===========================================================================

def pc_instrument_one_step(x, y, W1, b1, W2, b2, lr, wd,
                           n_inference_iters=15, inf_lr=0.1):
    """Run one PC training step with MemTracker to measure ARD.

    Key insight: each inference iteration and weight update only touches
    LOCAL buffers (one layer's weights + its neighbors' activations/errors).
    This should give much smaller reuse distances than backprop.
    """
    hidden = len(W1)
    n_input = len(x)
    tracker = MemTracker()

    # Initial buffer writes (parameters already in memory)
    tracker.write('W1', hidden * n_input)
    tracker.write('b1', hidden)
    tracker.write('W2', hidden)
    tracker.write('b2', 1)
    tracker.write('x', n_input)
    tracker.write('y', 1)

    # === 1. Forward init to get value node guesses ===
    tracker.read('x', n_input)
    tracker.read('W1', hidden * n_input)
    tracker.read('b1', hidden)
    mu1_pre = [sum(W1[j][i] * x[i] for i in range(n_input)) + b1[j]
               for j in range(hidden)]
    mu1 = [relu(v) for v in mu1_pre]
    tracker.write('mu1_pre', hidden)
    tracker.write('mu1', hidden)

    tracker.read('mu1', hidden)
    tracker.read('W2', hidden)
    tracker.read('b2', 1)
    mu2 = sum(W2[j] * mu1[j] for j in range(hidden)) + b2
    tracker.write('mu2', 1)

    # === 2. Inference phase ===
    # Each iteration only touches local buffers
    for t in range(n_inference_iters):
        # Compute pred0 = W1^T @ mu1 (generative prediction of x)
        # Needs: W1, mu1 (local to layer 1)
        tracker.read('W1', hidden * n_input)
        tracker.read('mu1', hidden)
        pred0 = [sum(W1[j][i] * mu1[j] for j in range(hidden))
                 for i in range(n_input)]

        # e0 = x - pred0
        tracker.read('x', n_input)
        e0 = [x[i] - pred0[i] for i in range(n_input)]
        tracker.write('e0', n_input)

        # pred1 = W2 * mu2 (prediction of hidden from output)
        tracker.read('W2', hidden)
        tracker.read('mu2', 1)
        pred1 = [W2[j] * mu2 for j in range(hidden)]

        # e1 = mu1 - pred1
        tracker.read('mu1', hidden)
        e1 = [mu1[j] - pred1[j] for j in range(hidden)]
        tracker.write('e1', hidden)

        # e2 = y - mu2
        tracker.read('y', 1)
        tracker.read('mu2', 1)
        e2 = y - mu2
        tracker.write('e2', 1)

        # Update mu1 using local errors
        tracker.read('e1', hidden)
        tracker.read('e0', n_input)
        tracker.read('W1', hidden * n_input)
        tracker.read('e2', 1)
        tracker.read('W2', hidden)
        tracker.read('mu1_pre', hidden)

        grad_from_e0 = [sum(W1[j][i] * e0[i] for i in range(n_input))
                        for j in range(hidden)]
        grad_from_e2 = [W2[j] * e2 for j in range(hidden)]

        for j in range(hidden):
            d = (-e1[j] + grad_from_e0[j] + grad_from_e2[j]) * relu_deriv(mu1_pre[j])
            mu1_pre[j] += inf_lr * d
            mu1[j] = relu(mu1_pre[j])

        tracker.write('mu1_pre', hidden)
        tracker.write('mu1', hidden)

        # Clamp mu2 = y
        mu2 = y
        tracker.write('mu2', 1)

    # === 3. Weight update phase (local rules) ===

    # Update W1: needs e0, mu1 (layer 0-1 boundary)
    tracker.read('e0', n_input)
    tracker.read('mu1', hidden)
    tracker.read('W1', hidden * n_input)
    for j in range(hidden):
        for i in range(n_input):
            W1[j][i] += lr * e0[i] * mu1[j] - wd * W1[j][i]
    tracker.write('W1', hidden * n_input)

    # Update b1: needs e1 (layer 1 local)
    tracker.read('e1', hidden)
    tracker.read('b1', hidden)
    for j in range(hidden):
        b1[j] += lr * e1[j] - wd * b1[j]
    tracker.write('b1', hidden)

    # Update W2: needs e2, mu1 (layer 1-2 boundary)
    tracker.read('e2', 1)
    tracker.read('mu1', hidden)
    tracker.read('W2', hidden)
    for j in range(hidden):
        W2[j] += lr * e2 * mu1[j] - wd * W2[j]
    tracker.write('W2', hidden)

    # Update b2: needs e2 (layer 2 local)
    tracker.read('e2', 1)
    tracker.read('b2', 1)
    b2 += lr * e2 - wd * b2
    tracker.write('b2', 1)

    return tracker


# ===========================================================================
# Backprop baseline for ARD comparison
# ===========================================================================

def run_backprop_baseline(config, verbose=True):
    """Run standard backprop and measure ARD on one step for comparison."""
    x_train, y_train, x_test, y_test, secret = generate(config)
    W1, b1, W2, b2 = init_params(config)

    best_acc = 0.0
    converge_epoch = None
    for epoch in range(1, config.max_epochs + 1):
        for i in range(len(x_train)):
            out, h_pre, h = forward(x_train[i], W1, b1, W2, b2)
            backward_and_update(x_train[i], y_train[i], out, h_pre, h,
                                W1, b1, W2, b2, config)

        outs = [forward(xt, W1, b1, W2, b2)[0] for xt in x_test]
        acc = accuracy(outs, y_test)
        if acc > best_acc:
            best_acc = acc
        if best_acc >= 0.9 and converge_epoch is None:
            converge_epoch = epoch
        if best_acc >= 1.0:
            break

    # Instrument one step for ARD
    tracker = MemTracker()
    tracker.write('W1', config.hidden * config.n_bits)
    tracker.write('b1', config.hidden)
    tracker.write('W2', config.hidden)
    tracker.write('b2', 1)
    tracker.write('x', config.n_bits)
    tracker.write('y', 1)

    out, h_pre, h = forward(x_train[0], W1, b1, W2, b2, tracker=tracker)
    backward_and_update(x_train[0], y_train[0], out, h_pre, h,
                        W1, b1, W2, b2, config, tracker=tracker)

    if verbose:
        print(f"    Backprop: acc={best_acc:.3f}, epoch={converge_epoch}")
        tracker.report()

    return {
        'best_test_acc': best_acc,
        'converged_epoch': converge_epoch,
        'ard': tracker.to_json(),
    }


# ===========================================================================
# Experiment runner
# ===========================================================================

def run_pc_experiment(n_bits, k_sparse, hidden, n_train, n_test,
                      lr, wd, n_inference_iters, inf_lr,
                      max_epochs, seed=42, label="", verbose=True):
    """Run predictive coding experiment with given parameters."""
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Predictive Coding: {label}")
        print(f"  n={n_bits}, k={k_sparse}, hidden={hidden}, lr={lr}, wd={wd}")
        print(f"  inference_iters={n_inference_iters}, inf_lr={inf_lr}")
        print(f"{'=' * 70}")

    config = Config(n_bits=n_bits, k_sparse=k_sparse, n_train=n_train,
                    n_test=n_test, hidden=hidden, seed=seed)
    x_train, y_train, x_test, y_test, secret = generate(config)

    if verbose:
        print(f"  Secret indices: {secret}")

    # Initialize PC network
    W1, b1, W2, b2 = pc_init_params(n_bits, hidden, seed=seed)

    start = time.time()
    best_test_acc = 0.0
    best_train_acc = 0.0
    converge_epoch = None

    for epoch in range(1, max_epochs + 1):
        for i in range(len(x_train)):
            W1, b1, W2, b2 = pc_train_step(
                x_train[i], y_train[i], W1, b1, W2, b2,
                lr, wd, n_inference_iters, inf_lr
            )

        # Evaluate
        train_acc = pc_accuracy(x_train, y_train, W1, b1, W2, b2)
        test_acc = pc_accuracy(x_test, y_test, W1, b1, W2, b2)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if best_test_acc >= 0.9 and converge_epoch is None:
            converge_epoch = epoch

        if verbose and (epoch <= 5 or epoch % 10 == 0 or test_acc > 0.8):
            elapsed = time.time() - start
            print(f"    Epoch {epoch:4d} | train={train_acc:.3f} test={test_acc:.3f} | {elapsed:.1f}s")

        if best_test_acc >= 1.0:
            break

        # Runtime guard
        if time.time() - start > 120:
            if verbose:
                print(f"    [Timeout at epoch {epoch}]")
            break

    elapsed = time.time() - start

    # Measure ARD on one step
    if verbose:
        print(f"\n  Instrumenting one PC step with MemTracker...")
    tracker = pc_instrument_one_step(
        x_train[0], y_train[0], W1, b1, W2, b2,
        lr, wd, n_inference_iters, inf_lr
    )
    if verbose:
        tracker.report()

    return {
        'label': label,
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'hidden': hidden,
        'n_train': n_train,
        'n_test': n_test,
        'lr': lr,
        'wd': wd,
        'n_inference_iters': n_inference_iters,
        'inf_lr': inf_lr,
        'max_epochs': max_epochs,
        'seed': seed,
        'best_train_acc': best_train_acc,
        'best_test_acc': best_test_acc,
        'converge_epoch': converge_epoch,
        'elapsed_s': round(elapsed, 3),
        'ard': tracker.to_json(),
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: Predictive Coding for Sparse Parity")
    print("  Comparing local PC learning vs global backprop (accuracy & ARD)")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # =================================================================
    # Config 1: Sanity check -- n=3, k=3
    # =================================================================
    print("\n" + "=" * 70)
    print("  CONFIG 1: Sanity check (n=3, k=3)")
    print("=" * 70)

    pc_3bit_results = []
    for seed in seeds:
        r = run_pc_experiment(
            n_bits=3, k_sparse=3, hidden=100,
            n_train=50, n_test=50,
            lr=0.01, wd=0.001,
            n_inference_iters=15, inf_lr=0.1,
            max_epochs=100, seed=seed,
            label=f"PC n=3/k=3 seed={seed}"
        )
        pc_3bit_results.append(r)

    all_results['pc_3bit'] = pc_3bit_results

    # Backprop baseline for n=3/k=3
    print(f"\n  --- Backprop baseline n=3/k=3 ---")
    bp_3bit_results = []
    for seed in seeds:
        config = Config(n_bits=3, k_sparse=3, n_train=50, n_test=50,
                        hidden=100, lr=0.5, wd=0.01, max_epochs=50, seed=seed)
        bp_r = run_backprop_baseline(config, verbose=True)
        bp_r['seed'] = seed
        bp_3bit_results.append(bp_r)
    all_results['bp_3bit'] = bp_3bit_results

    # =================================================================
    # Config 2: Main experiment -- n=20, k=3
    # =================================================================
    print("\n\n" + "=" * 70)
    print("  CONFIG 2: Main experiment (n=20, k=3, hidden=1000)")
    print("=" * 70)

    pc_20bit_results = []
    for seed in seeds:
        r = run_pc_experiment(
            n_bits=20, k_sparse=3, hidden=1000,
            n_train=500, n_test=200,
            lr=0.1, wd=0.01,
            n_inference_iters=15, inf_lr=0.05,
            max_epochs=50, seed=seed,
            label=f"PC n=20/k=3 seed={seed}"
        )
        pc_20bit_results.append(r)

    all_results['pc_20bit'] = pc_20bit_results

    # Backprop baseline for n=20/k=3
    print(f"\n  --- Backprop baseline n=20/k=3 ---")
    bp_20bit_results = []
    for seed in seeds:
        config = Config(n_bits=20, k_sparse=3, n_train=500, n_test=200,
                        hidden=1000, lr=0.1, wd=0.01, max_epochs=50, seed=seed)
        bp_r = run_backprop_baseline(config, verbose=True)
        bp_r['seed'] = seed
        bp_20bit_results.append(bp_r)
    all_results['bp_20bit'] = bp_20bit_results

    # =================================================================
    # Summary
    # =================================================================
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    print(f"  {'Method':<30} {'Config':>10} {'Acc':>8} {'ARD':>12} {'Epoch':>6}")
    print(f"  {'─'*30} {'─'*10} {'─'*8} {'─'*12} {'─'*6}")

    def summarize(results_list, method_name, config_str):
        accs = [r['best_test_acc'] for r in results_list]
        ards = [r['ard']['weighted_ard'] for r in results_list]
        epochs = [r.get('converge_epoch') or r.get('converged_epoch') or '---' for r in results_list]
        avg_acc = sum(accs) / len(accs)
        avg_ard = sum(ards) / len(ards)
        epoch_str = '/'.join(str(e) for e in epochs)
        print(f"  {method_name:<30} {config_str:>10} {avg_acc:>8.3f} {avg_ard:>12,.0f} {epoch_str:>6}")
        return avg_acc, avg_ard

    pc_3_acc, pc_3_ard = summarize(pc_3bit_results, 'Predictive Coding', 'n=3/k=3')
    bp_3_acc, bp_3_ard = summarize(bp_3bit_results, 'Backprop', 'n=3/k=3')
    pc_20_acc, pc_20_ard = summarize(pc_20bit_results, 'Predictive Coding', 'n=20/k=3')
    bp_20_acc, bp_20_ard = summarize(bp_20bit_results, 'Backprop', 'n=20/k=3')

    print(f"\n  ARD Ratios:")
    if bp_3_ard > 0 and pc_3_ard > 0:
        ratio_3 = bp_3_ard / pc_3_ard
        print(f"    n=3/k=3:  backprop/PC = {ratio_3:.2f}x", end="")
        if ratio_3 > 1:
            print(f" -> PC has {ratio_3:.1f}x LOWER ARD")
        else:
            print(f" -> Backprop has {1/ratio_3:.1f}x LOWER ARD")

    if bp_20_ard > 0 and pc_20_ard > 0:
        ratio_20 = bp_20_ard / pc_20_ard
        print(f"    n=20/k=3: backprop/PC = {ratio_20:.2f}x", end="")
        if ratio_20 > 1:
            print(f" -> PC has {ratio_20:.1f}x LOWER ARD")
        else:
            print(f" -> Backprop has {1/ratio_20:.1f}x LOWER ARD")

    print("=" * 90)

    # =================================================================
    # Save results
    # =================================================================
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_predictive_coding'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'results.json'

    save_data = {
        'experiment': 'exp_predictive_coding',
        'description': 'Predictive coding network for sparse parity -- local learning vs backprop',
        'hypothesis': 'PC local learning yields lower ARD than backprop due to smaller working sets per layer update',
        'approach': 'Predictive Coding (Rao & Ballard 1999, Millidge et al. 2021)',
        'configs': {
            'n3_k3': {
                'pc': pc_3bit_results,
                'bp': bp_3bit_results,
                'summary': {
                    'pc_avg_acc': pc_3_acc, 'bp_avg_acc': bp_3_acc,
                    'pc_avg_ard': pc_3_ard, 'bp_avg_ard': bp_3_ard,
                    'ard_ratio': bp_3_ard / pc_3_ard if pc_3_ard > 0 else None,
                },
            },
            'n20_k3': {
                'pc': pc_20bit_results,
                'bp': bp_20bit_results,
                'summary': {
                    'pc_avg_acc': pc_20_acc, 'bp_avg_acc': bp_20_acc,
                    'pc_avg_ard': pc_20_ard, 'bp_avg_ard': bp_20_ard,
                    'ard_ratio': bp_20_ard / pc_20_ard if pc_20_ard > 0 else None,
                },
            },
        },
    }

    save_json(save_data, results_path)
    print(f"\n  Results saved to: {results_path}")

    return all_results


if __name__ == '__main__':
    main()

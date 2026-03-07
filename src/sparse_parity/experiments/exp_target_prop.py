#!/usr/bin/env python3
"""
Experiment: Target Propagation for Sparse Parity

Approach #10 from research plan: Bengio (2014), Lee et al. (2015).

Each layer gets a local target computed by approximate inverse mappings
from the layer above. Layers train with local losses toward their targets.
Unlike Forward-Forward's greedy approach, targets carry information from
the global loss. After target computation (one backward pass of inverses),
all weight updates are local.

Architecture:
  - 2-layer network: n -> hidden (ReLU) -> 1 (linear)
  - For each layer, learn a forward mapping f_i and an approximate inverse g_i
  - Forward pass: h1 = relu(W1 @ x + b1), y_hat = W2 @ h1 + b2
  - Output target: t2 = y (the true label)
  - Backward inverse: t1 = g2(t2) (learned linear inverse of layer 2)
  - Local loss per layer: ||f_i(input_i) - t_i||^2
  - Inverse g_i trained alongside: ||g_i(f_i(input_i)) - input_i||^2

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_target_prop.py
"""

import sys
import math
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparse_parity.tracker import MemTracker


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n_bits, k_sparse, n_samples, seed=42):
    """Generate sparse parity data. Returns x, y, secret."""
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1).reshape(-1, 1)
    return x, y, secret


# =============================================================================
# TARGET PROPAGATION NETWORK
# =============================================================================

class TargetPropNet:
    """2-layer network with target propagation.

    Forward mappings:
        h1 = relu(W1 @ x + b1)          (layer 1)
        y_hat = W2 @ h1 + b2            (layer 2, linear output)

    Inverse mappings (learned):
        g2: R^1 -> R^hidden             (inverse of layer 2)
        g1: R^hidden -> R^n             (inverse of layer 1, not used for updates)

    Target computation:
        t_output = y (true label)
        t1 = g2(t_output)  -- target for hidden layer

    Local losses:
        L_layer2 = ||y_hat - t_output||^2
        L_layer1 = ||h1 - t1||^2

    Inverse training loss:
        L_inv2 = ||g2(W2 @ h1 + b2) - h1||^2
    """

    def __init__(self, n_input, hidden, seed=42):
        rng = np.random.RandomState(seed + 100)

        # Forward mapping layer 1: n_input -> hidden
        std1 = math.sqrt(2.0 / n_input)
        self.W1 = rng.randn(hidden, n_input) * std1
        self.b1 = np.zeros(hidden)

        # Forward mapping layer 2: hidden -> 1
        std2 = math.sqrt(2.0 / hidden)
        self.W2 = rng.randn(1, hidden) * std2
        self.b2 = np.zeros(1)

        # Inverse mapping g2: 1 -> hidden (maps output space back to hidden)
        self.G2 = rng.randn(hidden, 1) * 0.01
        self.g2_bias = np.zeros(hidden)

        self.n_input = n_input
        self.hidden = hidden

    def forward(self, x):
        """Forward pass. x is (n_input,). Returns y_hat, h1_pre, h1."""
        h1_pre = self.W1 @ x + self.b1
        h1 = np.maximum(h1_pre, 0.0)  # ReLU
        y_hat = self.W2 @ h1 + self.b2
        return y_hat, h1_pre, h1

    def inverse_g2(self, t_output):
        """Inverse mapping for layer 2: maps output target to hidden target."""
        return self.G2 @ t_output + self.g2_bias

    def compute_targets(self, y_true):
        """Compute local targets for each layer.

        t_output = y_true
        t1 = g2(y_true)
        """
        t_output = y_true  # shape (1,)
        t1 = self.inverse_g2(t_output)  # shape (hidden,)
        return t_output, t1

    def _clip_grad(self, g, max_norm=1.0):
        """Clip gradient vector/matrix by global norm."""
        norm = np.sqrt(np.sum(g ** 2))
        if norm > max_norm:
            g = g * (max_norm / norm)
        return g

    def train_step(self, x, y_true, lr_forward, lr_inverse, tracker=None):
        """One training step with target propagation.

        Steps:
        1. Forward pass to get activations
        2. Compute targets via inverse mappings
        3. Train inverse mappings (reconstruction loss)
        4. Update forward weights using local target losses
        """
        # --- Forward pass ---
        if tracker:
            tracker.read('x', self.n_input)
            tracker.read('W1', self.hidden * self.n_input)
            tracker.read('b1', self.hidden)

        h1_pre = self.W1 @ x + self.b1
        h1 = np.maximum(h1_pre, 0.0)

        if tracker:
            tracker.write('h1', self.hidden)
            tracker.read('h1', self.hidden)
            tracker.read('W2', self.hidden)
            tracker.read('b2', 1)

        y_hat = self.W2 @ h1 + self.b2

        if tracker:
            tracker.write('y_hat', 1)

        # --- Compute targets ---
        t_output = y_true  # shape (1,)

        if tracker:
            tracker.read('G2', self.hidden)
            tracker.read('g2_bias', self.hidden)

        t1 = self.inverse_g2(t_output)  # target for hidden layer

        if tracker:
            tracker.write('t1', self.hidden)

        # --- Train inverse mapping g2 ---
        # Reconstruction loss: ||g2(y_hat) - h1||^2
        # g2(y_hat) = G2 @ y_hat + g2_bias
        if tracker:
            tracker.read('G2', self.hidden)
            tracker.read('g2_bias', self.hidden)
            tracker.read('y_hat', 1)

        g2_recon = self.G2 @ y_hat + self.g2_bias  # (hidden,)

        if tracker:
            tracker.read('h1', self.hidden)

        inv_error = g2_recon - h1  # (hidden,)

        # Gradients for G2: d/dG2 ||G2 @ y_hat + g2_bias - h1||^2
        # = 2 * (G2 @ y_hat + g2_bias - h1) @ y_hat^T
        dG2 = self._clip_grad(2.0 * np.outer(inv_error, y_hat))  # (hidden, 1)
        dg2_bias = self._clip_grad(2.0 * inv_error)  # (hidden,)

        self.G2 -= lr_inverse * dG2
        self.g2_bias -= lr_inverse * dg2_bias

        if tracker:
            tracker.write('G2', self.hidden)
            tracker.write('g2_bias', self.hidden)

        # --- Update layer 2 (forward weights) using local target loss ---
        # L2 = ||y_hat - t_output||^2
        # d/dW2 = 2 * (y_hat - t_output) @ h1^T
        # d/db2 = 2 * (y_hat - t_output)
        if tracker:
            tracker.read('y_hat', 1)
            tracker.read('h1', self.hidden)
            tracker.read('W2', self.hidden)
            tracker.read('b2', 1)

        output_error = y_hat - t_output  # (1,)
        dW2 = self._clip_grad(2.0 * np.outer(output_error, h1))  # (1, hidden)
        db2 = self._clip_grad(2.0 * output_error)  # (1,)

        self.W2 -= lr_forward * dW2
        self.b2 -= lr_forward * db2

        if tracker:
            tracker.write('W2', self.hidden)
            tracker.write('b2', 1)

        # --- Update layer 1 (forward weights) using local target loss ---
        # L1 = ||h1 - t1||^2
        # But h1 = relu(W1 @ x + b1), so we need d(relu)/d(pre)
        # d/dW1 = 2 * (h1 - t1) * relu'(h1_pre) @ x^T
        # d/db1 = 2 * (h1 - t1) * relu'(h1_pre)
        if tracker:
            tracker.read('h1', self.hidden)
            tracker.read('t1', self.hidden)
            tracker.read('x', self.n_input)
            tracker.read('W1', self.hidden * self.n_input)
            tracker.read('b1', self.hidden)

        hidden_error = h1 - t1  # (hidden,)
        relu_mask = (h1_pre > 0).astype(float)  # (hidden,)
        layer1_grad = 2.0 * hidden_error * relu_mask  # (hidden,)

        dW1 = self._clip_grad(np.outer(layer1_grad, x))  # (hidden, n_input)
        db1 = self._clip_grad(layer1_grad)  # (hidden,)

        self.W1 -= lr_forward * dW1
        self.b1 -= lr_forward * db1

        if tracker:
            tracker.write('W1', self.hidden * self.n_input)
            tracker.write('b1', self.hidden)

        # Return losses for monitoring
        loss_output = float(np.sum(output_error ** 2))
        loss_hidden = float(np.sum(hidden_error ** 2))
        loss_inverse = float(np.sum(inv_error ** 2))

        return y_hat, loss_output, loss_hidden, loss_inverse

    def predict(self, x):
        """Predict label for a single input. Returns sign of output."""
        y_hat, _, _ = self.forward(x)
        return 1.0 if y_hat[0] >= 0 else -1.0

    def accuracy(self, X, Y):
        """Compute accuracy over a dataset."""
        correct = 0
        for i in range(len(X)):
            pred = self.predict(X[i])
            if pred == Y[i, 0]:
                correct += 1
        return correct / len(X)


# =============================================================================
# BACKPROP BASELINE (for ARD comparison)
# =============================================================================

class BackpropNet:
    """Standard backprop 2-layer network for comparison."""

    def __init__(self, n_input, hidden, seed=42):
        rng = np.random.RandomState(seed + 100)
        std1 = math.sqrt(2.0 / n_input)
        self.W1 = rng.randn(hidden, n_input) * std1
        self.b1 = np.zeros(hidden)
        std2 = math.sqrt(2.0 / hidden)
        self.W2 = rng.randn(1, hidden) * std2
        self.b2 = np.zeros(1)
        self.n_input = n_input
        self.hidden = hidden

    def forward(self, x):
        h1_pre = self.W1 @ x + self.b1
        h1 = np.maximum(h1_pre, 0.0)
        y_hat = self.W2 @ h1 + self.b2
        return y_hat, h1_pre, h1

    def _clip_grad(self, g, max_norm=1.0):
        """Clip gradient vector/matrix by global norm."""
        norm = np.sqrt(np.sum(g ** 2))
        if norm > max_norm:
            g = g * (max_norm / norm)
        return g

    def train_step(self, x, y_true, lr, tracker=None):
        if tracker:
            tracker.read('x', self.n_input)
            tracker.read('W1', self.hidden * self.n_input)
            tracker.read('b1', self.hidden)

        h1_pre = self.W1 @ x + self.b1
        h1 = np.maximum(h1_pre, 0.0)

        if tracker:
            tracker.write('h1_pre', self.hidden)
            tracker.write('h1', self.hidden)
            tracker.read('h1', self.hidden)
            tracker.read('W2', self.hidden)
            tracker.read('b2', 1)

        y_hat = self.W2 @ h1 + self.b2

        if tracker:
            tracker.write('y_hat', 1)

        # MSE loss backward
        error = y_hat - y_true  # (1,)

        if tracker:
            tracker.read('y_hat', 1)
            tracker.read('h1', self.hidden)

        dW2 = self._clip_grad(2.0 * np.outer(error, h1))
        db2 = self._clip_grad(2.0 * error)

        if tracker:
            tracker.write('dW2', self.hidden)
            tracker.write('db2', 1)
            tracker.read('W2', self.hidden)

        # Backprop through layer 2
        dh1 = (self.W2.T @ error).flatten()

        if tracker:
            tracker.write('dh1', self.hidden)
            tracker.read('dh1', self.hidden)
            tracker.read('h1_pre', self.hidden)

        # ReLU backward
        dh1_pre = dh1 * (h1_pre > 0).astype(float)

        if tracker:
            tracker.write('dh1_pre', self.hidden)
            tracker.read('dh1_pre', self.hidden)
            tracker.read('x', self.n_input)

        dW1 = self._clip_grad(np.outer(dh1_pre, x))
        db1 = self._clip_grad(dh1_pre)

        if tracker:
            tracker.read('W1', self.hidden * self.n_input)

        # Update all weights
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        if tracker:
            tracker.write('W2', self.hidden)
            tracker.write('b2', 1)
            tracker.write('W1', self.hidden * self.n_input)
            tracker.write('b1', self.hidden)

        return y_hat, float(np.sum(error ** 2))

    def predict(self, x):
        y_hat, _, _ = self.forward(x)
        return 1.0 if y_hat[0] >= 0 else -1.0

    def accuracy(self, X, Y):
        correct = 0
        for i in range(len(X)):
            pred = self.predict(X[i])
            if pred == Y[i, 0]:
                correct += 1
        return correct / len(X)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(n_bits, k_sparse, hidden, n_train, n_test, lr_forward,
                   lr_inverse, max_epochs, seed=42, label="", timeout=120):
    """Run target propagation experiment with given config."""
    print(f"\n{'=' * 70}")
    print(f"  Target Propagation: {label}")
    print(f"  n={n_bits}, k={k_sparse}, hidden={hidden}, lr_fwd={lr_forward}, lr_inv={lr_inverse}")
    print(f"{'=' * 70}")

    x_train, y_train, secret = generate_data(n_bits, k_sparse, n_train, seed=seed)
    x_test, y_test, _ = generate_data(n_bits, k_sparse, n_test, seed=seed + 1000)

    print(f"  Secret indices: {secret}")

    net = TargetPropNet(n_bits, hidden, seed=seed)

    start = time.time()
    best_test_acc = 0.0
    best_train_acc = 0.0
    converge_epoch = None
    history = []

    for epoch in range(1, max_epochs + 1):
        total_loss_out = 0.0
        total_loss_hid = 0.0
        total_loss_inv = 0.0

        for i in range(n_train):
            _, l_out, l_hid, l_inv = net.train_step(
                x_train[i], y_train[i], lr_forward, lr_inverse
            )
            total_loss_out += l_out
            total_loss_hid += l_hid
            total_loss_inv += l_inv

        train_acc = net.accuracy(x_train, y_train)
        test_acc = net.accuracy(x_test, y_test)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        entry = {
            'epoch': epoch,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'loss_output': total_loss_out / n_train,
            'loss_hidden': total_loss_hid / n_train,
            'loss_inverse': total_loss_inv / n_train,
        }
        history.append(entry)

        if epoch <= 5 or epoch % 10 == 0 or test_acc > 0.8:
            elapsed = time.time() - start
            print(f"    Epoch {epoch:4d} | train={train_acc:.3f} test={test_acc:.3f} | "
                  f"L_out={entry['loss_output']:.4f} L_hid={entry['loss_hidden']:.4f} "
                  f"L_inv={entry['loss_inverse']:.4f} | {elapsed:.1f}s")

        if best_test_acc >= 0.9 and converge_epoch is None:
            converge_epoch = epoch

        if best_test_acc >= 1.0:
            break

        if time.time() - start > timeout:
            print(f"    [Timeout at epoch {epoch}]")
            break

    elapsed = time.time() - start

    # --- ARD measurement ---
    print(f"\n  Instrumenting one target-prop step with MemTracker...")
    tracker = MemTracker()
    tracker.write('W1', hidden * n_bits)
    tracker.write('b1', hidden)
    tracker.write('W2', hidden)
    tracker.write('b2', 1)
    tracker.write('G2', hidden)
    tracker.write('g2_bias', hidden)
    tracker.write('x', n_bits)
    tracker.write('y', 1)

    net.train_step(x_train[0], y_train[0], lr_forward, lr_inverse, tracker=tracker)
    tracker.report()

    result = {
        'label': label,
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'hidden': hidden,
        'n_train': n_train,
        'n_test': n_test,
        'lr_forward': lr_forward,
        'lr_inverse': lr_inverse,
        'max_epochs': max_epochs,
        'seed': seed,
        'secret': secret,
        'best_train_acc': best_train_acc,
        'best_test_acc': best_test_acc,
        'converge_epoch': converge_epoch,
        'elapsed_s': round(elapsed, 3),
        'ard': tracker.to_json(),
        'final_losses': history[-1] if history else {},
    }

    return result


def run_backprop_baseline(n_bits, k_sparse, hidden, n_train, n_test,
                          lr, max_epochs, seed=42, label="", timeout=120):
    """Run backprop baseline for comparison."""
    print(f"\n{'=' * 70}")
    print(f"  Backprop Baseline: {label}")
    print(f"  n={n_bits}, k={k_sparse}, hidden={hidden}, lr={lr}")
    print(f"{'=' * 70}")

    x_train, y_train, secret = generate_data(n_bits, k_sparse, n_train, seed=seed)
    x_test, y_test, _ = generate_data(n_bits, k_sparse, n_test, seed=seed + 1000)

    print(f"  Secret indices: {secret}")

    net = BackpropNet(n_bits, hidden, seed=seed)

    start = time.time()
    best_test_acc = 0.0
    converge_epoch = None

    for epoch in range(1, max_epochs + 1):
        total_loss = 0.0
        for i in range(n_train):
            _, loss = net.train_step(x_train[i], y_train[i], lr)
            total_loss += loss

        train_acc = net.accuracy(x_train, y_train)
        test_acc = net.accuracy(x_test, y_test)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if epoch <= 5 or epoch % 10 == 0 or test_acc > 0.8:
            elapsed = time.time() - start
            print(f"    Epoch {epoch:4d} | train={train_acc:.3f} test={test_acc:.3f} | "
                  f"loss={total_loss/n_train:.4f} | {elapsed:.1f}s")

        if best_test_acc >= 0.9 and converge_epoch is None:
            converge_epoch = epoch

        if best_test_acc >= 1.0:
            break

        if time.time() - start > timeout:
            print(f"    [Timeout at epoch {epoch}]")
            break

    elapsed = time.time() - start

    # ARD measurement
    print(f"\n  Instrumenting one backprop step with MemTracker...")
    tracker = MemTracker()
    tracker.write('W1', hidden * n_bits)
    tracker.write('b1', hidden)
    tracker.write('W2', hidden)
    tracker.write('b2', 1)
    tracker.write('x', n_bits)
    tracker.write('y', 1)

    net.train_step(x_train[0], y_train[0], lr, tracker=tracker)
    tracker.report()

    return {
        'label': label,
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'hidden': hidden,
        'best_test_acc': best_test_acc,
        'converge_epoch': converge_epoch,
        'elapsed_s': round(elapsed, 3),
        'ard': tracker.to_json(),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: Target Propagation for Sparse Parity")
    print("  Approach: Bengio (2014), Lee et al. (2015)")
    print("  Each layer trains with local loss toward targets from inverses")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # =================================================================
    # Config 1: n=3, k=3 (sanity check)
    # =================================================================
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=3, k=3 (sanity check)")
    print("=" * 70)

    tp_3bit_results = []
    for seed in seeds:
        r = run_experiment(
            n_bits=3, k_sparse=3, hidden=100,
            n_train=50, n_test=50,
            lr_forward=0.005, lr_inverse=0.005,
            max_epochs=500, seed=seed,
            label=f"n=3/k=3 seed={seed}", timeout=60
        )
        tp_3bit_results.append(r)

    all_results['tp_3bit'] = tp_3bit_results

    # =================================================================
    # Config 2: n=20, k=3 (main test)
    # =================================================================
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=20, k=3 (main test)")
    print("=" * 70)

    tp_20bit_results = []
    for seed in seeds:
        r = run_experiment(
            n_bits=20, k_sparse=3, hidden=1000,
            n_train=500, n_test=200,
            lr_forward=0.01, lr_inverse=0.01,
            max_epochs=200, seed=seed,
            label=f"n=20/k=3 seed={seed}", timeout=120
        )
        tp_20bit_results.append(r)

    all_results['tp_20bit'] = tp_20bit_results

    # =================================================================
    # Backprop baselines (same configs, for ARD comparison)
    # =================================================================
    print("\n" + "=" * 70)
    print("  BACKPROP BASELINES")
    print("=" * 70)

    bp_3bit = run_backprop_baseline(
        n_bits=3, k_sparse=3, hidden=100,
        n_train=50, n_test=50,
        lr=0.01, max_epochs=500, seed=42,
        label="Backprop n=3/k=3", timeout=60
    )
    all_results['bp_3bit'] = bp_3bit

    bp_20bit = run_backprop_baseline(
        n_bits=20, k_sparse=3, hidden=1000,
        n_train=500, n_test=200,
        lr=0.01, max_epochs=200, seed=42,
        label="Backprop n=20/k=3", timeout=120
    )
    all_results['bp_20bit'] = bp_20bit

    # =================================================================
    # Summary
    # =================================================================
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n  {'Config':<30} {'Best Test Acc':>12} {'Converge':>10} {'ARD':>12} {'Time':>8}")
    print(f"  {'---'*25}")

    # Target prop results
    for config_name, results_list in [('tp_3bit', tp_3bit_results), ('tp_20bit', tp_20bit_results)]:
        accs = [r['best_test_acc'] for r in results_list]
        epochs = [r['converge_epoch'] for r in results_list]
        ards = [r['ard']['weighted_ard'] for r in results_list]
        times = [r['elapsed_s'] for r in results_list]

        avg_acc = np.mean(accs)
        conv_str = f"{np.mean([e for e in epochs if e is not None]):.0f}" if any(e is not None for e in epochs) else "N/A"
        avg_ard = np.mean(ards)
        avg_time = np.mean(times)

        label = f"TargetProp {config_name.replace('tp_', '')}"
        print(f"  {label:<30} {avg_acc:>12.3f} {conv_str:>10} {avg_ard:>12,.0f} {avg_time:>7.1f}s")

    # Backprop results
    for key in ['bp_3bit', 'bp_20bit']:
        r = all_results[key]
        label = f"Backprop {key.replace('bp_', '')}"
        conv_str = str(r['converge_epoch']) if r['converge_epoch'] else "N/A"
        print(f"  {label:<30} {r['best_test_acc']:>12.3f} {conv_str:>10} {r['ard']['weighted_ard']:>12,.0f} {r['elapsed_s']:>7.1f}s")

    # ARD comparison
    print(f"\n  ARD Comparison (single-step memory reuse distance):")
    for bits, tp_key, bp_key in [('3bit', 'tp_3bit', 'bp_3bit'), ('20bit', 'tp_20bit', 'bp_20bit')]:
        tp_ard = np.mean([r['ard']['weighted_ard'] for r in all_results[tp_key]])
        bp_ard = all_results[bp_key]['ard']['weighted_ard']
        if bp_ard > 0 and tp_ard > 0:
            ratio = bp_ard / tp_ard
            print(f"    {bits}: TargetProp ARD={tp_ard:,.0f}, Backprop ARD={bp_ard:,.0f}, ratio={ratio:.2f}x")

    print("=" * 70)

    # =================================================================
    # Save results
    # =================================================================
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_target_prop'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_target_prop',
            'description': 'Target Propagation for sparse parity (Bengio 2014, Lee et al. 2015)',
            'approach': 'Each layer gets local target from approximate inverse; weight updates are local',
            'configs': all_results,
        }, f, indent=2, default=convert_numpy)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

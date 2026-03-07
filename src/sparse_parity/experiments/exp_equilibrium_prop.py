#!/usr/bin/env python3
"""
Experiment: Equilibrium Propagation for Sparse Parity

Approach #9 from Scellier & Bengio (2017).
The network settles to an equilibrium (free phase), then is nudged toward
the correct output (clamped phase). Weight updates use the difference between
free and clamped states. Only forward passes, no backprop. Proven to
approximate backprop gradients in the limit of small nudging (beta -> 0).

Architecture: 2-layer energy-based network
  Layer 1: n_bits -> hidden (tanh activation)
  Layer 2: hidden -> 1 (tanh activation, output)

Energy: E = -0.5 * sum(s_i * W_ij * s_j) for connected pairs
         + cost(output, target) when clamped

Free phase:  run network to equilibrium (~20-50 iterative relaxation steps)
Clamped phase: add beta * cost term, re-settle (~20-50 steps)
Weight update: dW = (1/beta) * (s_i_clamped * s_j_clamped - s_i_free * s_j_free)

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_equilibrium_prop.py
"""

import sys
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
    y = np.prod(x[:, secret], axis=1)  # +1 or -1
    return x, y, secret


# =============================================================================
# EQUILIBRIUM PROPAGATION NETWORK
# =============================================================================

class EqPropNetwork:
    """
    2-layer energy-based network for equilibrium propagation.

    Layers:
      input (n_bits) -> hidden (n_hidden, tanh) -> output (1, tanh)

    State variables: s_hidden, s_output (continuous, clamped via tanh)
    Parameters: W1 (n_bits x n_hidden), b1 (n_hidden),
                W2 (n_hidden x 1), b2 (1)
    """

    def __init__(self, n_bits, n_hidden, seed=42):
        self.n_bits = n_bits
        self.n_hidden = n_hidden
        rng = np.random.RandomState(seed)

        # He-like initialization scaled for tanh
        std1 = np.sqrt(1.0 / n_bits)
        self.W1 = rng.randn(n_bits, n_hidden) * std1
        self.b1 = np.zeros(n_hidden)

        std2 = np.sqrt(1.0 / n_hidden)
        self.W2 = rng.randn(n_hidden, 1) * std2
        self.b2 = np.zeros(1)

    def energy(self, x, s_h, s_o):
        """
        Compute the energy of the network state.
        E = -x^T W1 s_h - b1^T s_h - s_h^T W2 s_o - b2^T s_o
            + 0.5 * sum(rho^{-1}(s_h_i)^2) + 0.5 * sum(rho^{-1}(s_o_j)^2)

        For tanh activation, the primitive is the Legendre transform term.
        We use a simplified energy: just the bilinear terms.
        """
        term1 = -x.dot(self.W1).dot(s_h)
        term2 = -self.b1.dot(s_h)
        term3 = -s_h.dot(self.W2).flatten()[0] * s_o[0]
        term4 = -self.b2[0] * s_o[0]
        return term1 + term2 + term3 + term4

    def cost(self, s_o, target):
        """Squared error cost: 0.5 * (s_o - target)^2."""
        return 0.5 * (s_o[0] - target) ** 2

    def free_phase(self, x, n_steps=30, step_size=0.5):
        """
        Run the network to equilibrium without any target nudging.
        Uses iterative relaxation: each unit updates toward the gradient
        of the energy w.r.t. its pre-activation, passed through tanh.

        Returns (s_hidden, s_output) at equilibrium.
        """
        # Initialize states to zero
        s_h = np.zeros(self.n_hidden)
        s_o = np.zeros(1)

        for _ in range(n_steps):
            # Update hidden units: s_h = tanh(x^T W1 + b1 + s_o * W2^T)
            pre_h = x.dot(self.W1) + self.b1 + s_o[0] * self.W2.T.flatten()
            s_h_new = np.tanh(pre_h)
            s_h = (1 - step_size) * s_h + step_size * s_h_new

            # Update output unit: s_o = tanh(s_h^T W2 + b2)
            pre_o = s_h.dot(self.W2).flatten() + self.b2
            s_o_new = np.tanh(pre_o)
            s_o = (1 - step_size) * s_o + step_size * s_o_new

        return s_h, s_o

    def clamped_phase(self, x, target, beta, s_h_init, s_o_init,
                      n_steps=30, step_size=0.5):
        """
        Run the network to a new equilibrium with target nudging.
        The cost gradient nudges the output toward the target.

        The update for s_o includes: -beta * d(cost)/d(s_o) = -beta * (s_o - target)
        This is added to the pre-activation before tanh.

        Returns (s_hidden, s_output) at clamped equilibrium.
        """
        s_h = s_h_init.copy()
        s_o = s_o_init.copy()

        for _ in range(n_steps):
            # Update hidden units (same as free phase, hidden aren't directly nudged)
            pre_h = x.dot(self.W1) + self.b1 + s_o[0] * self.W2.T.flatten()
            s_h_new = np.tanh(pre_h)
            s_h = (1 - step_size) * s_h + step_size * s_h_new

            # Update output with nudging: add -beta * (s_o - target) to drive toward target
            pre_o = s_h.dot(self.W2).flatten() + self.b2
            # Nudge term: gradient of -beta*cost w.r.t. pre-activation
            # d(-beta*cost)/d(s_o) = -beta*(s_o - target)
            # We add this to pre_o to nudge the equilibrium
            nudge = -beta * (s_o - target)
            s_o_new = np.tanh(pre_o + nudge)
            s_o = (1 - step_size) * s_o + step_size * s_o_new

        return s_h, s_o

    def update_weights(self, x, s_h_free, s_o_free, s_h_clamp, s_o_clamp,
                       beta, lr):
        """
        Equilibrium propagation weight update rule:
        dW = (1/beta) * (s_i_clamped * s_j_clamped - s_i_free * s_j_free)

        For W1: s_i = x, s_j = s_h
        For W2: s_i = s_h, s_j = s_o
        """
        inv_beta = 1.0 / beta

        # W1 update: dW1 = (1/beta) * (x^T s_h_clamp - x^T s_h_free)
        dW1 = inv_beta * (np.outer(x, s_h_clamp) - np.outer(x, s_h_free))
        self.W1 += lr * dW1

        # b1 update: db1 = (1/beta) * (s_h_clamp - s_h_free)
        db1 = inv_beta * (s_h_clamp - s_h_free)
        self.b1 += lr * db1

        # W2 update: dW2 = (1/beta) * (s_h_clamp * s_o_clamp - s_h_free * s_o_free)
        dW2 = inv_beta * (np.outer(s_h_clamp, s_o_clamp) -
                          np.outer(s_h_free, s_o_free))
        self.W2 += lr * dW2

        # b2 update: db2 = (1/beta) * (s_o_clamp - s_o_free)
        db2 = inv_beta * (s_o_clamp - s_o_free)
        self.b2 += lr * db2

    def predict(self, x, n_steps=30, step_size=0.5):
        """Predict by running free phase and returning sign of output."""
        _, s_o = self.free_phase(x, n_steps=n_steps, step_size=step_size)
        return 1.0 if s_o[0] >= 0 else -1.0

    def predict_batch(self, X, n_steps=30, step_size=0.5):
        """Predict for a batch of inputs."""
        return np.array([self.predict(x, n_steps, step_size) for x in X])


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(net, X, Y, beta, lr, free_steps, clamp_steps, step_size):
    """Train one epoch of equilibrium propagation. Returns mean cost."""
    total_cost = 0.0
    n = len(X)
    for i in range(n):
        x = X[i]
        target = Y[i]

        # Free phase
        s_h_free, s_o_free = net.free_phase(x, n_steps=free_steps,
                                            step_size=step_size)

        # Clamped phase (start from free equilibrium)
        s_h_clamp, s_o_clamp = net.clamped_phase(
            x, target, beta, s_h_free, s_o_free,
            n_steps=clamp_steps, step_size=step_size
        )

        # Weight update
        net.update_weights(x, s_h_free, s_o_free, s_h_clamp, s_o_clamp,
                           beta, lr)

        total_cost += net.cost(s_o_free, target)

    return total_cost / n


def evaluate(net, X, Y, free_steps=30, step_size=0.5):
    """Compute accuracy on dataset."""
    preds = net.predict_batch(X, n_steps=free_steps, step_size=step_size)
    correct = np.sum(preds == Y)
    return float(correct) / len(Y)


# =============================================================================
# ARD MEASUREMENT
# =============================================================================

def measure_ard(net, x, target, beta, lr, free_steps, clamp_steps, step_size):
    """
    Instrument one training step with MemTracker.
    Equilibrium propagation has two forward relaxation passes (free + clamped),
    no backward pass.
    """
    n_bits = net.n_bits
    n_hidden = net.n_hidden

    tracker = MemTracker()

    # Initial buffer writes (parameters and input in memory)
    tracker.write('W1', n_bits * n_hidden)
    tracker.write('b1', n_hidden)
    tracker.write('W2', n_hidden * 1)
    tracker.write('b2', 1)
    tracker.write('x', n_bits)
    tracker.write('target', 1)

    # === FREE PHASE (forward relaxation, n_steps iterations) ===
    for step in range(free_steps):
        # Read input and parameters
        tracker.read('x', n_bits)
        tracker.read('W1', n_bits * n_hidden)
        tracker.read('b1', n_hidden)
        tracker.read('W2', n_hidden * 1)

        # Read current hidden state (if exists)
        if step > 0:
            tracker.read('s_h', n_hidden)
            tracker.read('s_o', 1)

        # Compute new hidden and output states
        tracker.write('s_h', n_hidden)
        tracker.write('s_o', 1)

    # Save free-phase equilibrium states
    tracker.read('s_h', n_hidden)
    tracker.write('s_h_free', n_hidden)
    tracker.read('s_o', 1)
    tracker.write('s_o_free', 1)

    # === CLAMPED PHASE (forward relaxation with nudging, n_steps iterations) ===
    for step in range(clamp_steps):
        # Read input, parameters, and target
        tracker.read('x', n_bits)
        tracker.read('W1', n_bits * n_hidden)
        tracker.read('b1', n_hidden)
        tracker.read('W2', n_hidden * 1)
        tracker.read('target', 1)

        # Read current states
        tracker.read('s_h', n_hidden)
        tracker.read('s_o', 1)

        # Compute new states with nudging
        tracker.write('s_h', n_hidden)
        tracker.write('s_o', 1)

    # Save clamped-phase equilibrium states
    tracker.read('s_h', n_hidden)
    tracker.write('s_h_clamp', n_hidden)
    tracker.read('s_o', 1)
    tracker.write('s_o_clamp', 1)

    # === WEIGHT UPDATE ===
    # Read all four equilibrium states
    tracker.read('s_h_free', n_hidden)
    tracker.read('s_o_free', 1)
    tracker.read('s_h_clamp', n_hidden)
    tracker.read('s_o_clamp', 1)
    tracker.read('x', n_bits)

    # Read current weights, compute updates, write new weights
    tracker.read('W1', n_bits * n_hidden)
    tracker.write('W1', n_bits * n_hidden)
    tracker.read('b1', n_hidden)
    tracker.write('b1', n_hidden)
    tracker.read('W2', n_hidden * 1)
    tracker.write('W2', n_hidden * 1)
    tracker.read('b2', 1)
    tracker.write('b2', 1)

    return tracker


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(n_bits, k_sparse, n_hidden, n_train, n_test,
                   lr, beta, free_steps, clamp_steps, step_size,
                   max_epochs, seeds, label="", timeout_s=120):
    """Run equilibrium propagation experiment across multiple seeds."""
    print(f"\n{'=' * 70}")
    print(f"  Equilibrium Propagation: {label}")
    print(f"  n={n_bits}, k={k_sparse}, hidden={n_hidden}")
    print(f"  lr={lr}, beta={beta}, free_steps={free_steps}, clamp_steps={clamp_steps}")
    print(f"  n_train={n_train}, n_test={n_test}, max_epochs={max_epochs}")
    print(f"{'=' * 70}")

    seed_results = []

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        x_train, y_train, secret = generate_data(n_bits, k_sparse, n_train, seed=seed)
        x_test, y_test, _ = generate_data(n_bits, k_sparse, n_test, seed=seed + 1000)
        # Use same secret for test
        y_test = np.prod(x_test[:, secret], axis=1)

        print(f"  Secret indices: {secret}")

        net = EqPropNetwork(n_bits, n_hidden, seed=seed)

        start = time.time()
        best_train_acc = 0.0
        best_test_acc = 0.0
        converge_epoch = None
        history = []

        for epoch in range(1, max_epochs + 1):
            mean_cost = train_epoch(net, x_train, y_train, beta, lr,
                                    free_steps, clamp_steps, step_size)

            train_acc = evaluate(net, x_train, y_train, free_steps, step_size)
            test_acc = evaluate(net, x_test, y_test, free_steps, step_size)

            if train_acc > best_train_acc:
                best_train_acc = train_acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc

            if best_test_acc >= 0.9 and converge_epoch is None:
                converge_epoch = epoch

            history.append({
                'epoch': epoch,
                'cost': float(mean_cost),
                'train_acc': float(train_acc),
                'test_acc': float(test_acc),
            })

            if epoch <= 5 or epoch % 10 == 0 or test_acc > 0.8:
                elapsed = time.time() - start
                print(f"    Epoch {epoch:4d} | cost={mean_cost:.4f} | "
                      f"train={train_acc:.3f} test={test_acc:.3f} | {elapsed:.1f}s")

            if best_test_acc >= 1.0:
                print(f"    Perfect accuracy reached at epoch {epoch}")
                break

            if time.time() - start > timeout_s:
                print(f"    [Timeout at epoch {epoch} after {timeout_s}s]")
                break

        elapsed = time.time() - start

        # Measure ARD on one training step
        tracker = measure_ard(net, x_train[0], y_train[0], beta, lr,
                              free_steps, clamp_steps, step_size)
        tracker.report()

        seed_results.append({
            'seed': seed,
            'secret': secret,
            'best_train_acc': float(best_train_acc),
            'best_test_acc': float(best_test_acc),
            'converge_epoch': converge_epoch,
            'final_epoch': history[-1]['epoch'] if history else 0,
            'elapsed_s': round(elapsed, 3),
            'ard': tracker.to_json(),
            'history': history,
        })

        print(f"\n  Seed {seed}: best_train={best_train_acc:.3f}, "
              f"best_test={best_test_acc:.3f}, time={elapsed:.1f}s")

    return {
        'label': label,
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'n_hidden': n_hidden,
        'n_train': n_train,
        'n_test': n_test,
        'lr': lr,
        'beta': beta,
        'free_steps': free_steps,
        'clamp_steps': clamp_steps,
        'step_size': step_size,
        'max_epochs': max_epochs,
        'seeds': seed_results,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: Equilibrium Propagation for Sparse Parity")
    print("  Scellier & Bengio (2017) -- no backprop, only forward relaxation")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # -----------------------------------------------------------------
    # Config 1: n=3, k=3 (sanity check -- all bits are parity bits)
    # -----------------------------------------------------------------
    all_results['n3_k3'] = run_experiment(
        n_bits=3, k_sparse=3, n_hidden=100,
        n_train=50, n_test=50,
        lr=0.1, beta=0.2, free_steps=30, clamp_steps=30,
        step_size=0.5, max_epochs=200,
        seeds=seeds,
        label="n=3, k=3 (sanity check)",
        timeout_s=60,
    )

    # Check if sanity check passed; try different hyperparams if not
    best_3bit = max(r['best_test_acc'] for r in all_results['n3_k3']['seeds'])
    if best_3bit < 0.8:
        print("\n  Sanity check didn't reach 80%. Trying alternative hyperparams...")
        all_results['n3_k3_v2'] = run_experiment(
            n_bits=3, k_sparse=3, n_hidden=200,
            n_train=100, n_test=50,
            lr=0.05, beta=0.1, free_steps=50, clamp_steps=50,
            step_size=0.3, max_epochs=300,
            seeds=seeds,
            label="n=3, k=3 v2 (smaller beta, more steps)",
            timeout_s=60,
        )

    if best_3bit < 0.8:
        # Try with larger beta and different step size
        all_results['n3_k3_v3'] = run_experiment(
            n_bits=3, k_sparse=3, n_hidden=200,
            n_train=100, n_test=50,
            lr=0.2, beta=0.5, free_steps=40, clamp_steps=40,
            step_size=0.8, max_epochs=300,
            seeds=seeds,
            label="n=3, k=3 v3 (larger beta=0.5)",
            timeout_s=60,
        )

    # -----------------------------------------------------------------
    # Config 2: n=20, k=3 (the main challenge)
    # -----------------------------------------------------------------
    all_results['n20_k3'] = run_experiment(
        n_bits=20, k_sparse=3, n_hidden=1000,
        n_train=500, n_test=200,
        lr=0.05, beta=0.2, free_steps=30, clamp_steps=30,
        step_size=0.5, max_epochs=100,
        seeds=seeds,
        label="n=20, k=3 (hidden=1000)",
        timeout_s=180,
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    print(f"  {'Config':<30} | {'Seed':>4} | {'Train':>6} | {'Test':>6} | "
          f"{'Epoch':>5} | {'Time':>7} | {'ARD':>12}")
    print("  " + "-" * 88)

    for key, res in all_results.items():
        for sr in res['seeds']:
            ard_val = sr['ard'].get('weighted_ard', 0)
            conv = sr['converge_epoch'] if sr['converge_epoch'] else '-'
            print(f"  {res['label']:<30} | {sr['seed']:>4} | "
                  f"{sr['best_train_acc']:>6.3f} | {sr['best_test_acc']:>6.3f} | "
                  f"{str(conv):>5} | {sr['elapsed_s']:>6.1f}s | {ard_val:>12,.0f}")

    # Averages per config
    print("\n  AVERAGES:")
    print(f"  {'Config':<30} | {'Avg Train':>9} | {'Avg Test':>8} | "
          f"{'Avg Time':>8} | {'Avg ARD':>12}")
    print("  " + "-" * 75)
    for key, res in all_results.items():
        avg_train = np.mean([r['best_train_acc'] for r in res['seeds']])
        avg_test = np.mean([r['best_test_acc'] for r in res['seeds']])
        avg_time = np.mean([r['elapsed_s'] for r in res['seeds']])
        avg_ard = np.mean([r['ard'].get('weighted_ard', 0) for r in res['seeds']])
        print(f"  {res['label']:<30} | {avg_train:>9.3f} | {avg_test:>8.3f} | "
              f"{avg_time:>7.1f}s | {avg_ard:>12,.0f}")

    print("=" * 90)

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_equilibrium_prop'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_equilibrium_prop',
            'description': 'Equilibrium Propagation (Scellier & Bengio 2017) for sparse parity',
            'hypothesis': 'EP can solve sparse parity without backprop using only forward relaxation',
            'approach': 'energy-based network with free and clamped phases',
            'configs': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

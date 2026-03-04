"""
Experiment C: Per-layer forward-backward on 20-bit sparse parity.

Question: Does per-layer training converge on 20-bit (k=3)?
          What ARD improvement does it give vs standard backprop?

Method:
  - Config: n_bits=20, k_sparse=3, hidden=1000, LR=0.1, WD=0.01,
            n_train=500, n_test=200, single-sample SGD
  - Train with per-layer forward-backward (train_perlayer.py)
  - Train with standard backprop (train.py) for comparison
  - Track: test_accuracy, train_accuracy, ||w_t - w_0||_1
  - If converges (>90%), instrument one step with MemTracker
  - Print comparison table: method, accuracy, epochs_to_solve, ARD

Context: On 3-bit parity, per-layer gave 9.1% ARD improvement.
         exp1 showed winning config converges on 20-bit with mini-batch SGD.
"""

import sys
import time
import json
import copy
import math
import random
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params, forward, forward_batch
from sparse_parity.train import train, backward_and_update
from sparse_parity.train_perlayer import train_perlayer, train_step_perlayer, forward_batch_perlayer
from sparse_parity.tracker import MemTracker
from sparse_parity.metrics import hinge_loss, accuracy, save_json, timestamp


# ---------------------------------------------------------------------------
# Weight norm tracking
# ---------------------------------------------------------------------------

def weight_l1_movement(W1, b1, W2, b2, W1_0, b1_0, W2_0, b2_0):
    """Compute ||theta_t - theta_0||_1 for all parameters."""
    hidden = len(W1)
    n_bits = len(W1[0])
    total = 0.0
    for j in range(hidden):
        for i in range(n_bits):
            total += abs(W1[j][i] - W1_0[j][i])
        total += abs(b1[j] - b1_0[j])
        total += abs(W2[0][j] - W2_0[0][j])
    total += abs(b2[0] - b2_0[0])
    return total


# ---------------------------------------------------------------------------
# ARD measurement helpers
# ---------------------------------------------------------------------------

def measure_ard_standard(x, y, W1, b1, W2, b2, config):
    """Run one standard backprop step with MemTracker and return summary."""
    tracker = MemTracker()
    # Initial writes (parameters + input placed in memory)
    tracker.write('W1', config.hidden * config.n_bits)
    tracker.write('b1', config.hidden)
    tracker.write('W2', config.hidden)
    tracker.write('b2', 1)
    tracker.write('x', config.n_bits)
    tracker.write('y', 1)
    # Forward
    out, h_pre, h = forward(x, W1, b1, W2, b2, tracker=tracker)
    # Backward
    backward_and_update(x, y, out, h_pre, h, W1, b1, W2, b2, config, tracker=tracker)
    return tracker.summary()


def measure_ard_perlayer(x, y, W1, b1, W2, b2, config):
    """Run one per-layer step with MemTracker and return summary."""
    tracker = MemTracker()
    tracker.write('W1', config.hidden * config.n_bits)
    tracker.write('b1', config.hidden)
    tracker.write('W2', config.hidden)
    tracker.write('b2', 1)
    tracker.write('x', config.n_bits)
    tracker.write('y', 1)
    train_step_perlayer(x, y, W1, b1, W2, b2, config, tracker=tracker)
    return tracker.summary()


# ---------------------------------------------------------------------------
# Training loop with progress reporting + weight tracking
# ---------------------------------------------------------------------------

def train_with_tracking(method, x_train, y_train, x_test, y_test,
                        W1, b1, W2, b2, config, max_epochs,
                        W1_0, b1_0, W2_0, b2_0):
    """
    Train with either 'standard' or 'perlayer' method.
    Returns result dict with accs, losses, weight movements, timing.
    """
    train_accs, test_accs = [], []
    train_losses, test_losses = [], []
    weight_movements = []
    best_test_acc = 0.0
    solve_epoch = None
    total_steps = 0

    start = time.time()

    for epoch in range(1, max_epochs + 1):
        # --- One epoch of single-sample SGD ---
        for i in range(len(x_train)):
            if method == 'standard':
                out, h_pre, h = forward(x_train[i], W1, b1, W2, b2)
                backward_and_update(x_train[i], y_train[i], out, h_pre, h,
                                    W1, b1, W2, b2, config)
            else:
                train_step_perlayer(x_train[i], y_train[i],
                                    W1, b1, W2, b2, config)
            total_steps += 1

        # --- Evaluate ---
        if method == 'standard':
            tr_outs = forward_batch(x_train, W1, b1, W2, b2)
            te_outs = forward_batch(x_test, W1, b1, W2, b2)
        else:
            tr_outs = forward_batch_perlayer(x_train, W1, b1, W2, b2, config)
            te_outs = forward_batch_perlayer(x_test, W1, b1, W2, b2, config)

        tr_loss = hinge_loss(tr_outs, y_train)
        te_loss = hinge_loss(te_outs, y_test)
        tr_acc = accuracy(tr_outs, y_train)
        te_acc = accuracy(te_outs, y_test)
        wt_move = weight_l1_movement(W1, b1, W2, b2, W1_0, b1_0, W2_0, b2_0)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)
        weight_movements.append(wt_move)

        if te_acc > best_test_acc:
            best_test_acc = te_acc

        if te_acc >= 0.90 and solve_epoch is None:
            solve_epoch = epoch

        # Print every 10 epochs or on milestone
        if epoch % 10 == 0 or epoch == 1 or (te_acc >= 0.90 and epoch == solve_epoch):
            elapsed = time.time() - start
            print(f"    Epoch {epoch:4d} | train={tr_acc:.3f} test={te_acc:.3f} | "
                  f"loss={tr_loss:.4f} | ||w-w0||={wt_move:.2f} | {elapsed:.1f}s")

        # Early stop if solved
        if best_test_acc >= 0.99:
            elapsed = time.time() - start
            print(f"    *** SOLVED at epoch {epoch}! test_acc={te_acc:.3f} ({elapsed:.1f}s) ***")
            break

        # Time guard: if 1 epoch takes too long, bail early with a warning
        if epoch == 1:
            epoch1_time = time.time() - start
            if epoch1_time > 15.0:
                print(f"    WARNING: epoch 1 took {epoch1_time:.1f}s — will be very slow at hidden=1000")

    elapsed_total = time.time() - start

    return {
        'method': method,
        'best_test_acc': best_test_acc,
        'final_train_acc': train_accs[-1] if train_accs else 0,
        'final_test_acc': test_accs[-1] if test_accs else 0,
        'solve_epoch': solve_epoch,
        'epochs_run': len(train_accs),
        'total_steps': total_steps,
        'elapsed_s': elapsed_total,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'weight_movements': weight_movements,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    max_epochs = 200
    hidden = 1000

    config = Config(
        n_bits=20,
        k_sparse=3,
        n_train=500,
        n_test=200,
        hidden=hidden,
        lr=0.1,
        wd=0.01,
        max_epochs=max_epochs,
        seed=42,
    )

    print("=" * 70)
    print("  EXPERIMENT C: Per-Layer Forward-Backward on 20-bit Sparse Parity")
    print("=" * 70)
    print(f"  n_bits={config.n_bits}, k_sparse={config.k_sparse}, hidden={config.hidden}")
    print(f"  n_train={config.n_train}, n_test={config.n_test}")
    print(f"  lr={config.lr}, wd={config.wd}, single-sample SGD")
    print(f"  max_epochs={max_epochs}")
    print(f"  seed={config.seed}")
    print("=" * 70)

    # Generate data
    x_train, y_train, x_test, y_test, secret = generate(config)
    print(f"  Secret indices: {secret}")
    print(f"  Train balance: {sum(1 for y in y_train if y > 0)}/{len(y_train)} positive")
    print()

    # --- Time check: run 1 epoch of per-layer and see if hidden=1000 is feasible ---
    print("  Timing check (1 epoch per-layer)...")
    W1_tmp, b1_tmp, W2_tmp, b2_tmp = init_params(config)
    t0 = time.time()
    for i in range(len(x_train)):
        train_step_perlayer(x_train[i], y_train[i], W1_tmp, b1_tmp, W2_tmp, b2_tmp, config)
    t1 = time.time()
    epoch_time = t1 - t0
    print(f"  1 epoch = {epoch_time:.2f}s -> {max_epochs} epochs ~ {epoch_time * max_epochs:.0f}s")

    if epoch_time * max_epochs > 300:  # >5 min
        hidden = 500
        print(f"  ** Reducing hidden to {hidden} for feasibility **")
        config = Config(
            n_bits=20, k_sparse=3, n_train=500, n_test=200,
            hidden=hidden, lr=0.1, wd=0.01, max_epochs=max_epochs, seed=42,
        )
        # Regenerate data with same seed
        x_train, y_train, x_test, y_test, secret = generate(config)
    print()

    # ===================================================================
    # Run 1: Per-layer forward-backward
    # ===================================================================
    print("-" * 70)
    print("  [1/2] PER-LAYER FORWARD-BACKWARD")
    print("-" * 70)
    W1_pl, b1_pl, W2_pl, b2_pl = init_params(config)
    W1_0 = [row[:] for row in W1_pl]
    b1_0 = b1_pl[:]
    W2_0 = [row[:] for row in W2_pl]
    b2_0 = b2_pl[:]

    res_pl = train_with_tracking(
        'perlayer', x_train, y_train, x_test, y_test,
        W1_pl, b1_pl, W2_pl, b2_pl, config, max_epochs,
        W1_0, b1_0, W2_0, b2_0,
    )
    print()

    # ===================================================================
    # Run 2: Standard backprop (single-sample SGD, same init)
    # ===================================================================
    print("-" * 70)
    print("  [2/2] STANDARD BACKPROP (single-sample SGD)")
    print("-" * 70)
    W1_st, b1_st, W2_st, b2_st = init_params(config)
    W1_0s = [row[:] for row in W1_st]
    b1_0s = b1_st[:]
    W2_0s = [row[:] for row in W2_st]
    b2_0s = b2_st[:]

    res_st = train_with_tracking(
        'standard', x_train, y_train, x_test, y_test,
        W1_st, b1_st, W2_st, b2_st, config, max_epochs,
        W1_0s, b1_0s, W2_0s, b2_0s,
    )
    print()

    # ===================================================================
    # ARD measurement (if either method converged >90%)
    # ===================================================================
    ard_pl = None
    ard_st = None

    print("-" * 70)
    print("  ARD MEASUREMENT")
    print("-" * 70)

    # Find a sample with margin < 1 so the backward pass actually fires.
    # If all samples have margin >= 1, backward is a no-op and ARD is trivial.
    W1_m, b1_m, W2_m, b2_m = init_params(config)
    x_sample, y_sample = None, None
    for i in range(len(x_train)):
        out_check, _, _ = forward(x_train[i], W1_m, b1_m, W2_m, b2_m)
        if out_check * y_train[i] < 1.0:
            x_sample = x_train[i]
            y_sample = y_train[i]
            print(f"  Using sample {i} (margin={out_check * y_train[i]:.4f} < 1) for ARD measurement")
            break

    if x_sample is None:
        print("  WARNING: All samples have margin >= 1; backward pass never fires.")
        print("  ARD measurement would only show forward pass. Skipping.")
    else:
        # Standard backprop ARD (fresh init so backward fires)
        W1_a, b1_a, W2_a, b2_a = init_params(config)
        ard_st = measure_ard_standard(x_sample, y_sample, W1_a, b1_a, W2_a, b2_a, config)
        print(f"  Standard backprop ARD: {ard_st['weighted_ard']:,.0f} floats "
              f"(reads={ard_st['reads']}, writes={ard_st['writes']})")

        # Per-layer ARD (fresh init)
        W1_b, b1_b, W2_b, b2_b = init_params(config)
        ard_pl = measure_ard_perlayer(x_sample, y_sample, W1_b, b1_b, W2_b, b2_b, config)
        print(f"  Per-layer ARD:        {ard_pl['weighted_ard']:,.0f} floats "
              f"(reads={ard_pl['reads']}, writes={ard_pl['writes']})")

        if ard_st['weighted_ard'] > 0:
            improvement = (1 - ard_pl['weighted_ard'] / ard_st['weighted_ard']) * 100
            print(f"  ARD improvement:      {improvement:.1f}%")
    print()

    # ===================================================================
    # Comparison table
    # ===================================================================
    print("=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Method':<25} {'Best Acc':>10} {'Solve Ep':>10} {'Time':>8} {'ARD':>12}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*8} {'─'*12}")

    for label, res, ard in [
        ('Standard backprop', res_st, ard_st),
        ('Per-layer fwd-bwd', res_pl, ard_pl),
    ]:
        solve_str = str(res['solve_epoch']) if res['solve_epoch'] else 'N/A'
        ard_str = f"{ard['weighted_ard']:,.0f}" if ard else 'N/A'
        print(f"  {label:<25} {res['best_test_acc']:>10.3f} {solve_str:>10} "
              f"{res['elapsed_s']:>7.1f}s {ard_str:>12}")

    if ard_st and ard_pl and ard_st['weighted_ard'] > 0:
        imp = (1 - ard_pl['weighted_ard'] / ard_st['weighted_ard']) * 100
        print(f"\n  ARD improvement (per-layer vs standard): {imp:.1f}%")
    print("=" * 70)
    print()

    # ===================================================================
    # Verdict
    # ===================================================================
    converged_pl = res_pl['best_test_acc'] >= 0.90
    converged_st = res_st['best_test_acc'] >= 0.90
    print("  VERDICT:")
    print(f"    Per-layer converges on 20-bit: {'YES' if converged_pl else 'NO'} "
          f"(best={res_pl['best_test_acc']:.3f})")
    print(f"    Standard converges on 20-bit:  {'YES' if converged_st else 'NO'} "
          f"(best={res_st['best_test_acc']:.3f})")
    if ard_st and ard_pl and ard_st['weighted_ard'] > 0:
        imp = (1 - ard_pl['weighted_ard'] / ard_st['weighted_ard']) * 100
        print(f"    ARD improvement: {imp:.1f}%")
    print()

    # ===================================================================
    # Save results
    # ===================================================================
    results = {
        'experiment': 'exp_c_perlayer_20bit',
        'question': 'Does per-layer forward-backward converge on 20-bit sparse parity?',
        'config': {
            'n_bits': config.n_bits,
            'k_sparse': config.k_sparse,
            'n_train': config.n_train,
            'n_test': config.n_test,
            'hidden': config.hidden,
            'lr': config.lr,
            'wd': config.wd,
            'max_epochs': max_epochs,
            'seed': config.seed,
            'training': 'single-sample SGD',
        },
        'secret_indices': secret,
        'perlayer': {
            'best_test_acc': res_pl['best_test_acc'],
            'final_test_acc': res_pl['final_test_acc'],
            'solve_epoch': res_pl['solve_epoch'],
            'epochs_run': res_pl['epochs_run'],
            'elapsed_s': res_pl['elapsed_s'],
            'train_accs': res_pl['train_accs'],
            'test_accs': res_pl['test_accs'],
            'weight_movements': res_pl['weight_movements'],
            'ard': ard_pl,
        },
        'standard': {
            'best_test_acc': res_st['best_test_acc'],
            'final_test_acc': res_st['final_test_acc'],
            'solve_epoch': res_st['solve_epoch'],
            'epochs_run': res_st['epochs_run'],
            'elapsed_s': res_st['elapsed_s'],
            'train_accs': res_st['train_accs'],
            'test_accs': res_st['test_accs'],
            'weight_movements': res_st['weight_movements'],
            'ard': ard_st,
        },
    }

    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_c_perlayer_20bit'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'results.json'
    save_json(results, results_path)
    print(f"  Results saved to: {results_path}")


if __name__ == '__main__':
    main()

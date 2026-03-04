"""
Experiment E: Hinton's Forward-Forward algorithm for sparse parity.

Forward-Forward replaces backprop with two forward passes:
  - Positive pass: real data with correct labels -> increase goodness
  - Negative pass: real data with WRONG labels -> decrease goodness

Each layer learns locally using its own objective:
  Goodness = sum of squared ReLU activations

This should have much smaller ARD than backprop since:
  - No backward pass (no storing/reading intermediate activations for backprop)
  - Parameters accessed only locally per layer
  - No gradient chain across layers

Architecture:
  - Label embedding: prepend label (+1 or -1) as extra input dimension
  - Layer 1: (n_bits+1) -> hidden, ReLU, local FF objective
  - Layer 2: hidden -> hidden2, ReLU, local FF objective
  - Classification: for input x, run with label=+1 and label=-1,
    pick whichever gives higher total goodness

Reference: Hinton, "The Forward-Forward Algorithm" (2022)
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
from sparse_parity.metrics import accuracy, save_json, save_markdown, timestamp
from sparse_parity.tracker import MemTracker


# ===========================================================================
# Forward-Forward Implementation (pure Python)
# ===========================================================================

def ff_init_params(n_input, hidden1, hidden2, seed=42):
    """Initialize params for a 2-layer Forward-Forward network.

    Layer 1: n_input -> hidden1 (with ReLU)
    Layer 2: hidden1 -> hidden2 (with ReLU)

    Input includes label embedding, so n_input = n_bits + 1.
    """
    rng = random.Random(seed + 100)

    std1 = math.sqrt(2.0 / n_input)
    W1 = [[rng.gauss(0, std1) for _ in range(n_input)] for _ in range(hidden1)]
    b1 = [0.0] * hidden1

    std2 = math.sqrt(2.0 / hidden1)
    W2 = [[rng.gauss(0, std2) for _ in range(hidden1)] for _ in range(hidden2)]
    b2 = [0.0] * hidden2

    return W1, b1, W2, b2


def ff_layer_forward(x, W, b):
    """Single layer forward: h_pre = W*x + b, h = ReLU(h_pre).
    Returns (h_pre, h)."""
    n_out = len(W)
    n_in = len(x)
    h_pre = [sum(W[j][i] * x[i] for i in range(n_in)) + b[j] for j in range(n_out)]
    h = [max(0.0, v) for v in h_pre]
    return h_pre, h


def goodness(h):
    """Goodness = sum of squared activations (after ReLU)."""
    return sum(v * v for v in h)


def ff_layer_update(x, W, b, h_pre, h, lr, is_positive, threshold=2.0):
    """Update one layer using the Forward-Forward rule.

    For positive data: we want goodness > threshold
    For negative data: we want goodness < threshold

    Loss for positive: max(0, threshold - goodness)
    Loss for negative: max(0, goodness - threshold)

    d(goodness)/d(h_j) = 2 * h_j  (only if h_j > 0, i.e. ReLU active)
    d(h_j)/d(W_jk) = x_k  (if h_j > 0)
    d(h_j)/d(b_j) = 1     (if h_j > 0)

    So d(goodness)/d(W_jk) = 2 * h_j * x_k * (h_pre_j > 0)
       d(goodness)/d(b_j)  = 2 * h_j * (h_pre_j > 0)
    """
    g = goodness(h)
    n_out = len(W)
    n_in = len(x)

    if is_positive:
        # Want goodness > threshold. Loss = max(0, threshold - goodness)
        if g >= threshold:
            return g  # Already good, no update needed
        # Gradient direction: increase goodness -> follow gradient of goodness
        sign = 1.0
    else:
        # Want goodness < threshold. Loss = max(0, goodness - threshold)
        if g <= threshold:
            return g  # Already good, no update needed
        # Gradient direction: decrease goodness -> negate gradient of goodness
        sign = -1.0

    for j in range(n_out):
        if h_pre[j] <= 0:
            continue  # ReLU inactive, no gradient
        # d(goodness)/d(W_jk) = 2 * h_j * x_k
        grad_factor = sign * lr * 2.0 * h[j]
        for k in range(n_in):
            W[j][k] += grad_factor * x[k]
        b[j] += sign * lr * 2.0 * h[j]

    return g


def normalize_activations(h):
    """Normalize activations to unit length (as Hinton suggests).
    This prevents goodness from growing unboundedly."""
    norm = math.sqrt(sum(v * v for v in h) + 1e-8)
    return [v / norm for v in h]


def embed_label(x, label):
    """Prepend the label (+1 or -1) as the first input dimension."""
    return [label] + x


def ff_train_step(x, y, W1, b1, W2, b2, lr, threshold, tracker=None):
    """One Forward-Forward training step for a single sample.

    Two passes:
    1. Positive: embed correct label -> forward through both layers -> increase goodness
    2. Negative: embed wrong label -> forward through both layers -> decrease goodness
    """
    n_input = len(W1[0])
    hidden1 = len(W1)
    hidden2 = len(W2)

    # === POSITIVE PASS (correct label) ===
    x_pos = embed_label(x, y)

    if tracker:
        tracker.read('x', len(x))
        tracker.read('W1', hidden1 * n_input)
        tracker.read('b1', hidden1)

    h1_pre_pos, h1_pos = ff_layer_forward(x_pos, W1, b1)

    if tracker:
        tracker.write('h1_pos', hidden1)
        tracker.read('h1_pos', hidden1)

    # Update layer 1 for positive data
    g1_pos = ff_layer_update(x_pos, W1, b1, h1_pre_pos, h1_pos, lr, is_positive=True, threshold=threshold)

    if tracker:
        tracker.write('W1', hidden1 * n_input)
        tracker.write('b1', hidden1)

    # Normalize before feeding to next layer (Hinton's recommendation)
    h1_norm_pos = normalize_activations(h1_pos)

    if tracker:
        tracker.read('W2', hidden2 * hidden1)
        tracker.read('b2', hidden2)

    h2_pre_pos, h2_pos = ff_layer_forward(h1_norm_pos, W2, b2)

    if tracker:
        tracker.write('h2_pos', hidden2)
        tracker.read('h2_pos', hidden2)

    g2_pos = ff_layer_update(h1_norm_pos, W2, b2, h2_pre_pos, h2_pos, lr, is_positive=True, threshold=threshold)

    if tracker:
        tracker.write('W2', hidden2 * hidden1)
        tracker.write('b2', hidden2)

    # === NEGATIVE PASS (wrong label) ===
    x_neg = embed_label(x, -y)

    if tracker:
        tracker.read('x', len(x))
        tracker.read('W1', hidden1 * n_input)
        tracker.read('b1', hidden1)

    h1_pre_neg, h1_neg = ff_layer_forward(x_neg, W1, b1)

    if tracker:
        tracker.write('h1_neg', hidden1)
        tracker.read('h1_neg', hidden1)

    g1_neg = ff_layer_update(x_neg, W1, b1, h1_pre_neg, h1_neg, lr, is_positive=False, threshold=threshold)

    if tracker:
        tracker.write('W1', hidden1 * n_input)
        tracker.write('b1', hidden1)

    h1_norm_neg = normalize_activations(h1_neg)

    if tracker:
        tracker.read('W2', hidden2 * hidden1)
        tracker.read('b2', hidden2)

    h2_pre_neg, h2_neg = ff_layer_forward(h1_norm_neg, W2, b2)

    if tracker:
        tracker.write('h2_neg', hidden2)
        tracker.read('h2_neg', hidden2)

    g2_neg = ff_layer_update(h1_norm_neg, W2, b2, h2_pre_neg, h2_neg, lr, is_positive=False, threshold=threshold)

    if tracker:
        tracker.write('W2', hidden2 * hidden1)
        tracker.write('b2', hidden2)

    return g1_pos + g2_pos, g1_neg + g2_neg


def ff_predict(x, W1, b1, W2, b2):
    """Predict label by comparing goodness for label=+1 vs label=-1.

    Returns +1.0 or -1.0.
    """
    # Try label = +1
    x_pos = embed_label(x, 1.0)
    _, h1_pos = ff_layer_forward(x_pos, W1, b1)
    h1_norm_pos = normalize_activations(h1_pos)
    _, h2_pos = ff_layer_forward(h1_norm_pos, W2, b2)
    g_pos = goodness(h1_pos) + goodness(h2_pos)

    # Try label = -1
    x_neg = embed_label(x, -1.0)
    _, h1_neg = ff_layer_forward(x_neg, W1, b1)
    h1_norm_neg = normalize_activations(h1_neg)
    _, h2_neg = ff_layer_forward(h1_norm_neg, W2, b2)
    g_neg = goodness(h1_neg) + goodness(h2_neg)

    return 1.0 if g_pos >= g_neg else -1.0


def ff_predict_batch(xs, W1, b1, W2, b2):
    """Predict labels for a batch of inputs."""
    return [ff_predict(x, W1, b1, W2, b2) for x in xs]


def ff_accuracy(xs, ys, W1, b1, W2, b2):
    """Compute accuracy on a dataset."""
    preds = ff_predict_batch(xs, W1, b1, W2, b2)
    correct = sum(1 for p, y in zip(preds, ys) if p == y)
    return correct / len(ys)


# ===========================================================================
# Backprop baseline for ARD comparison
# ===========================================================================

def run_backprop_baseline(config):
    """Run standard backprop and measure ARD on one step for comparison."""
    x_train, y_train, x_test, y_test, secret = generate(config)
    W1, b1, W2, b2 = init_params(config)

    # Train until convergence or max epochs
    best_acc = 0.0
    for epoch in range(1, config.max_epochs + 1):
        for i in range(len(x_train)):
            out, h_pre, h = forward(x_train[i], W1, b1, W2, b2)
            backward_and_update(x_train[i], y_train[i], out, h_pre, h,
                                W1, b1, W2, b2, config)

        outs = [forward(xt, W1, b1, W2, b2)[0] for xt in x_test]
        acc = accuracy(outs, y_test)
        if acc > best_acc:
            best_acc = acc
        if best_acc >= 0.9:
            break

    # Instrument one step
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

    return {
        'best_test_acc': best_acc,
        'converged_epoch': epoch,
        'ard': tracker.to_json(),
    }


# ===========================================================================
# Forward-Forward ARD measurement
# ===========================================================================

def instrument_ff_step(x, y, W1, b1, W2, b2, lr, threshold):
    """Run one FF training step with MemTracker instrumentation."""
    n_input = len(W1[0])
    hidden1 = len(W1)
    hidden2 = len(W2)

    tracker = MemTracker()

    # Initial buffer writes
    tracker.write('W1', hidden1 * n_input)
    tracker.write('b1', hidden1)
    tracker.write('W2', hidden2 * hidden1)
    tracker.write('b2', hidden2)
    tracker.write('x', len(x))
    tracker.write('y', 1)

    ff_train_step(x, y, W1, b1, W2, b2, lr, threshold, tracker=tracker)

    return tracker


# ===========================================================================
# Experiment runner
# ===========================================================================

def run_ff_experiment(n_bits, k_sparse, hidden1, hidden2, n_train, n_test,
                      lr, threshold, max_epochs, seed=42, label=""):
    """Run Forward-Forward experiment with given parameters."""
    print(f"\n{'=' * 70}")
    print(f"  Forward-Forward: {label}")
    print(f"  n_bits={n_bits}, k_sparse={k_sparse}, hidden1={hidden1}, hidden2={hidden2}")
    print(f"  n_train={n_train}, n_test={n_test}, lr={lr}, threshold={threshold}")
    print(f"{'=' * 70}")

    # Generate data using the existing data module
    config = Config(n_bits=n_bits, k_sparse=k_sparse, n_train=n_train, n_test=n_test,
                    hidden=hidden1, seed=seed)
    x_train, y_train, x_test, y_test, secret = generate(config)

    print(f"  Secret indices: {secret}")

    # Initialize FF network (n_bits + 1 for label embedding)
    W1, b1, W2, b2 = ff_init_params(n_bits + 1, hidden1, hidden2, seed=seed)

    start = time.time()
    best_test_acc = 0.0
    best_train_acc = 0.0
    converge_epoch = None

    for epoch in range(1, max_epochs + 1):
        # Train
        total_g_pos = 0.0
        total_g_neg = 0.0
        for i in range(len(x_train)):
            g_pos, g_neg = ff_train_step(x_train[i], y_train[i], W1, b1, W2, b2,
                                          lr, threshold)
            total_g_pos += g_pos
            total_g_neg += g_neg

        # Evaluate
        train_acc = ff_accuracy(x_train, y_train, W1, b1, W2, b2)
        test_acc = ff_accuracy(x_test, y_test, W1, b1, W2, b2)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if epoch <= 5 or epoch % 10 == 0 or test_acc > 0.8:
            elapsed = time.time() - start
            avg_g_pos = total_g_pos / len(x_train)
            avg_g_neg = total_g_neg / len(x_train)
            print(f"    Epoch {epoch:4d} | train={train_acc:.3f} test={test_acc:.3f} | "
                  f"g_pos={avg_g_pos:.2f} g_neg={avg_g_neg:.2f} | {elapsed:.1f}s")

        if best_test_acc >= 0.9 and converge_epoch is None:
            converge_epoch = epoch

        # Early stopping if perfect
        if best_test_acc >= 1.0:
            break

        # Runtime guard: stop after 60 seconds for any single config
        if time.time() - start > 60:
            print(f"    [Timeout at epoch {epoch}]")
            break

    elapsed = time.time() - start

    # Measure ARD
    print(f"\n  Instrumenting one FF step with MemTracker...")
    tracker = instrument_ff_step(x_train[0], y_train[0], W1, b1, W2, b2, lr, threshold)
    tracker.report()

    result = {
        'label': label,
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'hidden1': hidden1,
        'hidden2': hidden2,
        'n_train': n_train,
        'n_test': n_test,
        'lr': lr,
        'threshold': threshold,
        'max_epochs': max_epochs,
        'best_train_acc': best_train_acc,
        'best_test_acc': best_test_acc,
        'converge_epoch': converge_epoch,
        'elapsed_s': elapsed,
        'ard': tracker.to_json(),
    }

    return result


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT E: Forward-Forward Algorithm for Sparse Parity")
    print("  Comparing against standard backprop for accuracy and ARD")
    print("=" * 70)

    results = {}

    # ----- 1. Forward-Forward on 3-bit parity (easy case) -----
    results['ff_3bit'] = run_ff_experiment(
        n_bits=3, k_sparse=3,
        hidden1=100, hidden2=100,
        n_train=50, n_test=50,
        lr=0.01, threshold=2.0,
        max_epochs=100,
        seed=42,
        label="3-bit parity (n=3, k=3)"
    )

    # If 3-bit didn't work well, try different hyperparameters
    if results['ff_3bit']['best_test_acc'] < 0.8:
        print("\n  3-bit basic config did not reach 80%. Trying with larger network & different lr...")
        results['ff_3bit_v2'] = run_ff_experiment(
            n_bits=3, k_sparse=3,
            hidden1=500, hidden2=500,
            n_train=100, n_test=100,
            lr=0.005, threshold=1.0,
            max_epochs=200,
            seed=42,
            label="3-bit parity v2 (larger, lower threshold)"
        )

    if results['ff_3bit']['best_test_acc'] < 0.8:
        # Try yet another variant with different threshold and lr
        results['ff_3bit_v3'] = run_ff_experiment(
            n_bits=3, k_sparse=3,
            hidden1=200, hidden2=200,
            n_train=100, n_test=100,
            lr=0.02, threshold=0.5,
            max_epochs=200,
            seed=42,
            label="3-bit parity v3 (threshold=0.5)"
        )

    # ----- 2. Try 20-bit if 3-bit worked -----
    best_3bit_acc = max(r['best_test_acc'] for k, r in results.items() if '3bit' in k)

    if best_3bit_acc >= 0.8:
        print("\n  3-bit solved! Scaling to 20-bit...")
        results['ff_20bit'] = run_ff_experiment(
            n_bits=20, k_sparse=3,
            hidden1=500, hidden2=500,
            n_train=500, n_test=200,
            lr=0.01, threshold=2.0,
            max_epochs=100,
            seed=42,
            label="20-bit parity (n=20, k=3)"
        )
    else:
        print(f"\n  3-bit best accuracy: {best_3bit_acc:.3f} -- skipping 20-bit.")
        # Still try 20-bit with a small run to get ARD comparison
        results['ff_20bit'] = run_ff_experiment(
            n_bits=20, k_sparse=3,
            hidden1=500, hidden2=500,
            n_train=500, n_test=200,
            lr=0.01, threshold=2.0,
            max_epochs=20,
            seed=42,
            label="20-bit parity (n=20, k=3) [limited epochs]"
        )

    # ----- 3. Backprop baseline for ARD comparison -----
    print(f"\n{'=' * 70}")
    print(f"  BACKPROP BASELINE (for ARD comparison)")
    print(f"{'=' * 70}")

    # 3-bit backprop
    print("\n  Running 3-bit backprop baseline...")
    bp_config_3 = Config(n_bits=3, k_sparse=3, n_train=50, n_test=50,
                         hidden=100, lr=0.5, wd=0.01, max_epochs=50, seed=42)
    bp_3bit = run_backprop_baseline(bp_config_3)
    results['bp_3bit'] = bp_3bit
    print(f"    Backprop 3-bit: acc={bp_3bit['best_test_acc']:.3f}, "
          f"ARD={bp_3bit['ard']['weighted_ard']:.0f}")

    # 20-bit backprop
    print("\n  Running 20-bit backprop baseline...")
    bp_config_20 = Config(n_bits=20, k_sparse=3, n_train=500, n_test=200,
                          hidden=500, lr=0.1, wd=0.01, max_epochs=100, seed=42)
    bp_20bit = run_backprop_baseline(bp_config_20)
    results['bp_20bit'] = bp_20bit
    print(f"    Backprop 20-bit: acc={bp_20bit['best_test_acc']:.3f}, "
          f"ARD={bp_20bit['ard']['weighted_ard']:.0f}")

    # ----- 4. Comparison summary -----
    print("\n\n" + "=" * 70)
    print("  ARD COMPARISON: Forward-Forward vs Backprop")
    print("=" * 70)
    print(f"  {'Method':<35} {'Bits':>5} {'Test Acc':>9} {'Weighted ARD':>14} {'Reads':>7} {'Writes':>7}")
    print(f"  {'─'*35} {'─'*5} {'─'*9} {'─'*14} {'─'*7} {'─'*7}")

    for key in sorted(results.keys()):
        r = results[key]
        ard_data = r.get('ard', {})
        n = r.get('n_bits', '?')
        acc = r.get('best_test_acc', 0)
        w_ard = ard_data.get('weighted_ard', 0)
        reads = ard_data.get('reads', 0)
        writes = ard_data.get('writes', 0)
        label = r.get('label', key)
        print(f"  {label:<35} {n:>5} {acc:>9.3f} {w_ard:>14,.0f} {reads:>7} {writes:>7}")

    # ARD ratio
    ff_3_ard = results['ff_3bit']['ard']['weighted_ard']
    bp_3_ard = results['bp_3bit']['ard']['weighted_ard']
    if bp_3_ard > 0 and ff_3_ard > 0:
        ratio = bp_3_ard / ff_3_ard
        print(f"\n  3-bit ARD ratio (backprop/FF): {ratio:.2f}x")
        if ratio > 1:
            print(f"  -> Forward-Forward has {ratio:.1f}x LOWER ARD (more energy-efficient per step)")
        else:
            print(f"  -> Backprop has {1/ratio:.1f}x LOWER ARD")

    ff_20_ard = results.get('ff_20bit', {}).get('ard', {}).get('weighted_ard', 0)
    bp_20_ard = results['bp_20bit']['ard']['weighted_ard']
    if bp_20_ard > 0 and ff_20_ard > 0:
        ratio = bp_20_ard / ff_20_ard
        print(f"\n  20-bit ARD ratio (backprop/FF): {ratio:.2f}x")
        if ratio > 1:
            print(f"  -> Forward-Forward has {ratio:.1f}x LOWER ARD (more energy-efficient per step)")
        else:
            print(f"  -> Backprop has {1/ratio:.1f}x LOWER ARD")

    print("=" * 70)

    # ----- 5. Save results -----
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_e_forward_forward'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    save_json(results, results_path)
    print(f"\n  Results saved to: {results_path}")

    return results


if __name__ == '__main__':
    main()

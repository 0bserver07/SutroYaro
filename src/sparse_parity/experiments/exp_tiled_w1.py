#!/usr/bin/env python3
"""
Experiment: Tiled W1 Updates for Sparse Parity

Hypothesis: W1 (input-to-hidden) dominates ARD at 75% of all float reads.
In standard backprop, W1 is read in forward, then read again much later in
backward -- the intervening operations evict it from L1 cache.

Tiling breaks W1 into blocks along the hidden dimension. For each tile:
  forward through that slice -> store partial hidden activation ->
  backward through that slice -> update W1 slice immediately.

Each tile of W1 is read in forward and re-read in backward while still
in L1 cache, drastically reducing reuse distance.

For hidden=1000, n=20: W1 is 20x1000 = 20,000 floats = 80KB (exceeds 64KB L1).
A tile of T=250: 20x250 = 5,000 floats = 20KB (fits L1).

Structure:
  Part 1: ARD measurement (1 step only, hidden=1000) -- fast
  Part 2: Accuracy verification (hidden=500, max_epochs=50) -- feasible in pure Python

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_tiled_w1.py
"""

import sys
import time
import json
import math
import random
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparse_parity.data import generate
from sparse_parity.config import Config
from sparse_parity.tracker import MemTracker
from sparse_parity.model import init_params, forward, forward_batch
from sparse_parity.train import backward_and_update
from sparse_parity.metrics import hinge_loss, accuracy


# =============================================================================
# TILED FORWARD-BACKWARD-UPDATE
# =============================================================================

def tiled_train_step(x, y, W1, b1, W2, b2, config, tile_size, tracker=None):
    """
    Tiled forward-backward for one sample.

    The math is identical to standard backprop -- we just reorder operations
    so that each W1 tile's forward read and backward read are close together
    in the memory access stream.

    Forward: compute h_pre and h tile-by-tile (each tile reads its W1 slice).
    Output layer: standard (reads full h, W2, b2).
    Backward output layer: standard (produces dh).
    Backward hidden layer: tile-by-tile, each tile reads its W1 slice for
    gradient computation and updates immediately.
    """
    hidden = len(W1)
    n_bits = len(x)
    n_tiles = (hidden + tile_size - 1) // tile_size

    # === FORWARD: tile-by-tile through W1 ===
    h_pre = [0.0] * hidden
    h = [0.0] * hidden

    for t in range(n_tiles):
        t_start = t * tile_size
        t_end = min(t_start + tile_size, hidden)
        t_size = t_end - t_start

        if tracker:
            tracker.read('x', n_bits)
            tracker.read(f'W1_tile{t}', t_size * n_bits)
            tracker.read(f'b1_tile{t}', t_size)

        for j in range(t_start, t_end):
            h_pre[j] = sum(W1[j][i] * x[i] for i in range(n_bits)) + b1[j]

        if tracker:
            tracker.write(f'h_pre_tile{t}', t_size)
            tracker.read(f'h_pre_tile{t}', t_size)

        for j in range(t_start, t_end):
            h[j] = max(0.0, h_pre[j])

        if tracker:
            tracker.write(f'h_tile{t}', t_size)

    # === FORWARD: output layer (uses full h) ===
    if tracker:
        for t in range(n_tiles):
            t_start = t * tile_size
            t_end = min(t_start + tile_size, hidden)
            t_size = t_end - t_start
            tracker.read(f'h_tile{t}', t_size)
        tracker.read('W2', hidden)
        tracker.read('b2', 1)

    out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]

    if tracker:
        tracker.write('out', 1)

    # === CHECK MARGIN ===
    if tracker:
        tracker.read('out', 1)
        tracker.read('y', 1)

    margin = out * y
    if margin >= 1.0:
        return out

    dout = -y

    if tracker:
        tracker.write('dout', 1)

    # === BACKWARD: output layer ===
    if tracker:
        tracker.read('dout', 1)
        for t in range(n_tiles):
            t_start = t * tile_size
            t_end = min(t_start + tile_size, hidden)
            t_size = t_end - t_start
            tracker.read(f'h_tile{t}', t_size)

    dW2_0 = [dout * h[j] for j in range(hidden)]
    db2_0 = dout

    if tracker:
        tracker.write('dW2', hidden)
        tracker.write('db2', 1)

    # dh = W2^T * dout
    if tracker:
        tracker.read('W2', hidden)
        tracker.read('dout', 1)

    dh = [W2[0][j] * dout for j in range(hidden)]

    if tracker:
        tracker.write('dh', hidden)

    # === BACKWARD: tile-by-tile through W1 ===
    for t in range(n_tiles):
        t_start = t * tile_size
        t_end = min(t_start + tile_size, hidden)
        t_size = t_end - t_start

        # ReLU backward for this tile
        if tracker:
            tracker.read(f'dh_tile{t}', t_size)
            tracker.read(f'h_pre_tile{t}', t_size)

        dh_pre_tile = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0)
                       for j in range(t_start, t_end)]

        if tracker:
            tracker.write(f'dh_pre_tile{t}', t_size)

        # W1 tile gradient + immediate update
        if tracker:
            tracker.read(f'dh_pre_tile{t}', t_size)
            tracker.read('x', n_bits)
            tracker.read(f'W1_tile{t}', t_size * n_bits)

        for jj, j in enumerate(range(t_start, t_end)):
            for i in range(n_bits):
                grad = dh_pre_tile[jj] * x[i]
                W1[j][i] -= config.lr * (grad + config.wd * W1[j][i])

        if tracker:
            tracker.write(f'W1_tile{t}', t_size * n_bits)

        # b1 tile update
        if tracker:
            tracker.read(f'dh_pre_tile{t}', t_size)
            tracker.read(f'b1_tile{t}', t_size)

        for jj, j in enumerate(range(t_start, t_end)):
            b1[j] -= config.lr * (dh_pre_tile[jj] + config.wd * b1[j])

        if tracker:
            tracker.write(f'b1_tile{t}', t_size)

    # === UPDATE: output layer ===
    if tracker:
        tracker.read('dW2', hidden)
        tracker.read('W2', hidden)

    for j in range(hidden):
        W2[0][j] -= config.lr * (dW2_0[j] + config.wd * W2[0][j])

    if tracker:
        tracker.write('W2', hidden)
        tracker.read('db2', 1)
        tracker.read('b2', 1)

    b2[0] -= config.lr * (db2_0 + config.wd * b2[0])

    if tracker:
        tracker.write('b2', 1)

    return out


def forward_only(x, W1, b1, W2, b2):
    """Standard forward pass for evaluation."""
    hidden = len(W1)
    n_bits = len(x)
    h_pre = [sum(W1[j][i] * x[i] for i in range(n_bits)) + b1[j] for j in range(hidden)]
    h = [max(0.0, v) for v in h_pre]
    out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]
    return out


def forward_batch_eval(xs, W1, b1, W2, b2):
    """Forward pass for batch evaluation."""
    return [forward_only(x, W1, b1, W2, b2) for x in xs]


# =============================================================================
# ARD MEASUREMENT (1 step only)
# =============================================================================

def measure_ard_baseline(config):
    """Measure baseline ARD for 1 step of standard backprop."""
    x_train, y_train, _, _, _ = generate(config)
    W1, b1, W2, b2 = init_params(config)

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

    return tracker.to_json()


def measure_ard_tiled(config, tile_size):
    """Measure tiled ARD for 1 step."""
    x_train, y_train, _, _, _ = generate(config)
    W1, b1, W2, b2 = init_params(config)

    tracker = MemTracker()
    n_tiles = (config.hidden + tile_size - 1) // tile_size
    for t in range(n_tiles):
        t_start = t * tile_size
        t_end = min(t_start + tile_size, config.hidden)
        t_size = t_end - t_start
        tracker.write(f'W1_tile{t}', t_size * config.n_bits)
        tracker.write(f'b1_tile{t}', t_size)
    tracker.write('W2', config.hidden)
    tracker.write('b2', 1)
    tracker.write('x', config.n_bits)
    tracker.write('y', 1)

    tiled_train_step(x_train[0], y_train[0], W1, b1, W2, b2,
                     config, tile_size, tracker=tracker)

    return tracker.to_json()


# =============================================================================
# ACCURACY VERIFICATION (smaller hidden for speed)
# =============================================================================

def verify_accuracy_baseline(config):
    """Train baseline and return accuracy metrics."""
    x_train, y_train, x_test, y_test, secret = generate(config)
    W1, b1, W2, b2 = init_params(config)

    best_test_acc = 0.0
    start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        for i in range(len(x_train)):
            out, h_pre, h = forward(x_train[i], W1, b1, W2, b2)
            backward_and_update(x_train[i], y_train[i], out, h_pre, h,
                                W1, b1, W2, b2, config)

        te_outs = forward_batch_eval(x_test, W1, b1, W2, b2)
        te_acc = accuracy(te_outs, y_test)
        if te_acc > best_test_acc:
            best_test_acc = te_acc
        if best_test_acc >= 1.0:
            break

    elapsed = time.time() - start
    return {
        'best_test_acc': best_test_acc,
        'epochs': epoch,
        'elapsed_s': round(elapsed, 3),
    }


def verify_accuracy_tiled(config, tile_size):
    """Train with tiling and return accuracy metrics."""
    x_train, y_train, x_test, y_test, secret = generate(config)
    W1, b1, W2, b2 = init_params(config)

    best_test_acc = 0.0
    start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        for i in range(len(x_train)):
            tiled_train_step(x_train[i], y_train[i], W1, b1, W2, b2,
                             config, tile_size)

        te_outs = forward_batch_eval(x_test, W1, b1, W2, b2)
        te_acc = accuracy(te_outs, y_test)
        if te_acc > best_test_acc:
            best_test_acc = te_acc
        if best_test_acc >= 1.0:
            break

    elapsed = time.time() - start
    return {
        'best_test_acc': best_test_acc,
        'epochs': epoch,
        'elapsed_s': round(elapsed, 3),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: Tiled W1 Updates for Sparse Parity")
    print("  Approach #11: Tile W1 to keep slices in L1 cache")
    print("=" * 70)

    tile_sizes = [50, 100, 250, 500]
    seeds = [42, 43, 44]
    n_bits = 20
    k_sparse = 3
    hidden_ard = 1000   # For ARD measurement
    hidden_acc = 500    # For accuracy verification (faster)
    max_epochs_acc = 50 # Enough to see grokking start

    all_results = {
        'experiment': 'exp_tiled_w1',
        'description': 'Tiled W1 updates to reduce ARD by keeping W1 tiles in L1 cache',
        'hypothesis': 'Tiling W1 along hidden dim reduces reuse distance for W1 reads',
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'hidden_ard': hidden_ard,
        'hidden_acc': hidden_acc,
        'tile_sizes': tile_sizes,
        'seeds': seeds,
        'w1_total_floats': hidden_ard * n_bits,
        'w1_total_bytes': hidden_ard * n_bits * 4,
    }

    # ===================================================================
    # PART 1: ARD MEASUREMENT (hidden=1000, 1 step only)
    # ===================================================================
    print(f"\n{'=' * 70}")
    print(f"  PART 1: ARD MEASUREMENT (hidden={hidden_ard}, 1 step)")
    print(f"{'=' * 70}")

    ard_results = {'baseline': [], 'tiled': {}}
    for ts in tile_sizes:
        ard_results['tiled'][f'T{ts}'] = []

    for seed in seeds:
        config_ard = Config(
            n_bits=n_bits, k_sparse=k_sparse, hidden=hidden_ard,
            lr=0.1, wd=0.01, max_epochs=1,
            n_train=500, n_test=200, seed=seed,
        )

        print(f"\n  Seed {seed}:")

        # Baseline ARD
        base_tracker = measure_ard_baseline(config_ard)
        base_ard = base_tracker['weighted_ard']
        print(f"    Baseline ARD: {base_ard:,.0f}")
        ard_results['baseline'].append({
            'seed': seed,
            'ard': base_ard,
            'tracker': base_tracker,
        })

        # Tiled ARD
        for ts in tile_sizes:
            n_tiles = (hidden_ard + ts - 1) // ts
            tile_kb = ts * n_bits * 4 / 1024
            tiled_tracker = measure_ard_tiled(config_ard, ts)
            tiled_ard = tiled_tracker['weighted_ard']
            change_pct = (tiled_ard - base_ard) / base_ard * 100 if base_ard > 0 else 0
            print(f"    Tiled T={ts:>3} ({n_tiles:>2} tiles, {tile_kb:>5.1f}KB): "
                  f"ARD={tiled_ard:>8,.0f}  ({change_pct:+.1f}%)")
            ard_results['tiled'][f'T{ts}'].append({
                'seed': seed,
                'tile_size': ts,
                'n_tiles': n_tiles,
                'tile_kb': round(tile_kb, 1),
                'ard': tiled_ard,
                'tracker': tiled_tracker,
            })

    all_results['ard_measurement'] = ard_results

    # ===================================================================
    # PART 2: ACCURACY VERIFICATION (hidden=500, max_epochs=50)
    # ===================================================================
    print(f"\n{'=' * 70}")
    print(f"  PART 2: ACCURACY VERIFICATION (hidden={hidden_acc}, "
          f"max_epochs={max_epochs_acc})")
    print(f"{'=' * 70}")

    acc_results = {'baseline': [], 'tiled': {}}
    for ts in tile_sizes:
        acc_results['tiled'][f'T{ts}'] = []

    for seed in seeds:
        config_acc = Config(
            n_bits=n_bits, k_sparse=k_sparse, hidden=hidden_acc,
            lr=0.1, wd=0.01, max_epochs=max_epochs_acc,
            n_train=500, n_test=200, seed=seed,
        )

        print(f"\n  Seed {seed}:")

        # Baseline accuracy
        t0 = time.time()
        base_acc_result = verify_accuracy_baseline(config_acc)
        print(f"    Baseline: acc={base_acc_result['best_test_acc']:.1%} "
              f"in {base_acc_result['epochs']} epochs ({base_acc_result['elapsed_s']:.1f}s)")
        acc_results['baseline'].append({
            'seed': seed,
            **base_acc_result,
        })

        # Tiled accuracy (test a subset of tile sizes for speed)
        for ts in tile_sizes:
            tiled_acc_result = verify_accuracy_tiled(config_acc, ts)
            print(f"    Tiled T={ts:>3}: acc={tiled_acc_result['best_test_acc']:.1%} "
                  f"in {tiled_acc_result['epochs']} epochs ({tiled_acc_result['elapsed_s']:.1f}s)")
            acc_results['tiled'][f'T{ts}'].append({
                'seed': seed,
                'tile_size': ts,
                **tiled_acc_result,
            })

    all_results['accuracy_verification'] = acc_results

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print(f"\n\n{'=' * 90}")
    print(f"  SUMMARY: ARD (hidden={hidden_ard})")
    print(f"{'=' * 90}")

    base_ards = [r['ard'] for r in ard_results['baseline']]
    avg_base_ard = sum(base_ards) / len(base_ards)

    header = (f"  {'Method':<25} {'Avg ARD':>12} {'ARD Change':>12} "
              f"{'Tile KB':>10} {'N Tiles':>8}")
    print(header)
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 8}")
    print(f"  {'Baseline':<25} {avg_base_ard:>12,.0f} {'---':>12} "
          f"{'N/A':>10} {'1':>8}")

    summary_rows = []
    for ts in tile_sizes:
        key = f'T{ts}'
        entries = ard_results['tiled'][key]
        ards = [r['ard'] for r in entries]
        avg_ard = sum(ards) / len(ards)
        tile_kb = entries[0]['tile_kb']
        n_tiles = entries[0]['n_tiles']
        change_pct = (avg_ard - avg_base_ard) / avg_base_ard * 100
        print(f"  {'Tiled T=' + str(ts):<25} {avg_ard:>12,.0f} {change_pct:>+11.1f}% "
              f"{tile_kb:>10.1f} {n_tiles:>8}")
        summary_rows.append({
            'tile_size': ts,
            'avg_ard': avg_ard,
            'change_pct': round(change_pct, 1),
            'tile_kb': tile_kb,
            'n_tiles': n_tiles,
        })

    print(f"\n{'=' * 90}")
    print(f"  SUMMARY: ACCURACY (hidden={hidden_acc})")
    print(f"{'=' * 90}")

    base_accs = [r['best_test_acc'] for r in acc_results['baseline']]
    avg_base_acc = sum(base_accs) / len(base_accs)
    print(f"  {'Method':<25} {'Avg Acc':>10} {'Avg Epochs':>12}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 12}")
    print(f"  {'Baseline':<25} {avg_base_acc:>10.1%} "
          f"{sum(r['epochs'] for r in acc_results['baseline'])/len(acc_results['baseline']):>12.0f}")

    for ts in tile_sizes:
        key = f'T{ts}'
        entries = acc_results['tiled'][key]
        accs = [r['best_test_acc'] for r in entries]
        avg_acc = sum(accs) / len(accs)
        avg_ep = sum(r['epochs'] for r in entries) / len(entries)
        print(f"  {'Tiled T=' + str(ts):<25} {avg_acc:>10.1%} {avg_ep:>12.0f}")

    print(f"{'=' * 90}")

    all_results['summary'] = {
        'avg_baseline_ard': avg_base_ard,
        'avg_baseline_acc': avg_base_acc,
        'tiled_ard_summary': summary_rows,
    }

    # ===================================================================
    # Per-buffer breakdown
    # ===================================================================
    print(f"\n  Per-buffer ARD breakdown (seed={seeds[0]}):")
    print(f"  {'-' * 70}")

    base_tracker = ard_results['baseline'][0].get('tracker', {})
    if base_tracker and 'per_buffer' in base_tracker:
        print(f"\n  BASELINE per-buffer:")
        print(f"  {'Buffer':<15} {'Size':>8} {'Reads':>6} {'Avg Dist':>12}")
        print(f"  {'-' * 15} {'-' * 8} {'-' * 6} {'-' * 12}")
        for name, info in sorted(base_tracker['per_buffer'].items()):
            print(f"  {name:<15} {info['size']:>8,} {info['read_count']:>6} "
                  f"{info['avg_dist']:>12,.0f}")

    # Best tiled variant
    best_ts = min(tile_sizes,
                  key=lambda ts: sum(r['ard'] for r in ard_results['tiled'][f'T{ts}']))
    best_tracker = ard_results['tiled'][f'T{best_ts}'][0].get('tracker', {})
    if best_tracker and 'per_buffer' in best_tracker:
        print(f"\n  BEST TILED (T={best_ts}) per-buffer:")
        print(f"  {'Buffer':<25} {'Size':>8} {'Reads':>6} {'Avg Dist':>12}")
        print(f"  {'-' * 25} {'-' * 8} {'-' * 6} {'-' * 12}")
        for name, info in sorted(best_tracker['per_buffer'].items()):
            print(f"  {name:<25} {info['size']:>8,} {info['read_count']:>6} "
                  f"{info['avg_dist']:>12,.0f}")

    # ===================================================================
    # Save results
    # ===================================================================
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_tiled_w1'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Compact the tracker data for JSON serialization
    save_results = copy.deepcopy(all_results)

    # Compress tracker data in ard_measurement
    for entry in save_results['ard_measurement']['baseline']:
        if entry.get('tracker'):
            t = entry['tracker']
            entry['tracker_summary'] = {
                'weighted_ard': t.get('weighted_ard', 0),
                'total_floats_accessed': t.get('total_floats_accessed', 0),
                'reads': t.get('reads', 0),
                'writes': t.get('writes', 0),
            }
            if 'per_buffer' in t:
                entry['tracker_per_buffer'] = {
                    name: {
                        'size': info['size'],
                        'avg_dist': round(info['avg_dist'], 1),
                        'read_count': info['read_count'],
                    }
                    for name, info in t['per_buffer'].items()
                }
            del entry['tracker']

    for key in save_results['ard_measurement']['tiled']:
        for entry in save_results['ard_measurement']['tiled'][key]:
            if entry.get('tracker'):
                t = entry['tracker']
                entry['tracker_summary'] = {
                    'weighted_ard': t.get('weighted_ard', 0),
                    'total_floats_accessed': t.get('total_floats_accessed', 0),
                    'reads': t.get('reads', 0),
                    'writes': t.get('writes', 0),
                }
                if 'per_buffer' in t:
                    entry['tracker_per_buffer'] = {
                        name: {
                            'size': info['size'],
                            'avg_dist': round(info['avg_dist'], 1),
                            'read_count': info['read_count'],
                        }
                        for name, info in t['per_buffer'].items()
                    }
                del entry['tracker']

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

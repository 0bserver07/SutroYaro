#!/usr/bin/env python3
"""
Experiment: Mutual Information Estimation for Sparse Parity

Hypothesis: MI(y; prod(x_S)) = log(2) (~0.693 bits) for the true subset S,
and ~0 for all wrong subsets. Unlike Fourier (which computes exact product
correlation), MI detects ANY nonlinear relationship. For binary parity this
is equivalent, but the framework generalizes.

Method: Exhaustive search over all C(n,k) subsets. For each subset S,
compute prod(x[:, S]) and measure MI with y using a 2x2 contingency table.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_mutual_info.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from itertools import combinations
from math import comb, log

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
    y = np.prod(x[:, secret], axis=1)
    return x, y, secret


# =============================================================================
# MUTUAL INFORMATION (from 2x2 contingency table)
# =============================================================================

def compute_mi_binary(a, b, n_samples):
    """
    Compute mutual information between two binary {-1, +1} vectors.

    Uses the 2x2 contingency table:
        MI = sum_{a_val, b_val} p(a_val, b_val) * log(p(a_val, b_val) / (p(a_val) * p(b_val)))

    For perfectly correlated binary variables, MI = log(2) ~ 0.693 nats.
    For independent binary variables, MI ~ 0.
    """
    # Count joint occurrences
    # Map -1 -> 0, +1 -> 1 for indexing
    a_idx = ((a + 1) / 2).astype(int)
    b_idx = ((b + 1) / 2).astype(int)

    # 2x2 contingency table
    joint = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            joint[i, j] = np.sum((a_idx == i) & (b_idx == j))

    # Normalize to probabilities
    joint_p = joint / n_samples

    # Marginals
    p_a = joint_p.sum(axis=1)
    p_b = joint_p.sum(axis=0)

    # MI = sum p(a,b) * log(p(a,b) / (p(a) * p(b)))
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if joint_p[i, j] > 0 and p_a[i] > 0 and p_b[j] > 0:
                mi += joint_p[i, j] * log(joint_p[i, j] / (p_a[i] * p_b[j]))

    return mi


# =============================================================================
# MI SOLVER
# =============================================================================

def mi_solve(x, y, n_bits, k_sparse, tracker=None):
    """
    Find the secret subset by exhaustive MI computation.

    For each k-subset S of {0..n-1}:
        parity_S = product(x[:, S], axis=1)
        MI_S = MI(y, parity_S)

    The true subset has MI ~= log(2) = 0.693 nats, all others ~= 0.

    Returns (predicted_secret, best_mi, n_subsets, all_mi_values).
    """
    n_samples = x.shape[0]

    if tracker:
        tracker.write('x', x.size)
        tracker.write('y', n_samples)

    best_mi = -1.0
    best_subset = None
    n_subsets = comb(n_bits, k_sparse)

    # Track top-5 and distribution stats
    mi_values = []

    for subset in combinations(range(n_bits), k_sparse):
        if tracker:
            tracker.read('x', n_samples * k_sparse)  # read k columns
            tracker.read('y', n_samples)

        parity = np.prod(x[:, list(subset)], axis=1)
        mi = compute_mi_binary(parity, y, n_samples)
        mi_values.append((list(subset), mi))

        if mi > best_mi:
            best_mi = mi
            best_subset = list(subset)

    return best_subset, best_mi, n_subsets, mi_values


# =============================================================================
# RUN ONE CONFIG
# =============================================================================

def run_config(n_bits, k_sparse, n_samples, seed=42, use_tracker=False, verbose=True):
    """Run MI solver on one config. Returns result dict."""
    n_subsets = comb(n_bits, k_sparse)

    if verbose:
        print(f"  n={n_bits}, k={k_sparse}: {n_subsets:,} subsets, {n_samples} samples")

    x, y, secret = generate_data(n_bits, k_sparse, n_samples, seed=seed)

    tracker = MemTracker() if use_tracker else None

    start = time.time()
    predicted, best_mi, _, mi_values = mi_solve(x, y, n_bits, k_sparse, tracker=tracker)
    elapsed = time.time() - start

    correct = (predicted == secret)

    # Test on fresh data
    rng_te = np.random.RandomState(seed + 1000)
    x_te = rng_te.choice([-1.0, 1.0], size=(500, n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    y_pred = np.prod(x_te[:, predicted], axis=1)
    te_acc = float(np.mean(y_pred == y_te))

    # MI distribution stats
    all_mi = [v for _, v in mi_values]
    mi_sorted = sorted(all_mi, reverse=True)
    top5 = [(s, round(m, 6)) for s, m in sorted(mi_values, key=lambda x: x[1], reverse=True)[:5]]

    # Wrong subset stats (exclude the true subset)
    wrong_mi = [m for s, m in mi_values if s != secret]
    max_wrong_mi = max(wrong_mi) if wrong_mi else 0.0
    mean_wrong_mi = float(np.mean(wrong_mi)) if wrong_mi else 0.0

    # Theoretical MI for perfectly correlated binary vars
    theoretical_mi = log(2)  # ~0.693 nats

    if verbose:
        status = "CORRECT" if correct else "WRONG"
        print(f"    Secret:    {secret}")
        print(f"    Predicted: {predicted} ({status})")
        print(f"    Best MI:   {best_mi:.6f} nats (theoretical max: {theoretical_mi:.6f})")
        print(f"    MI gap:    {best_mi - max_wrong_mi:.6f} (best - 2nd best)")
        print(f"    Wrong MI:  mean={mean_wrong_mi:.6f}, max={max_wrong_mi:.6f}")
        print(f"    Test acc:  {te_acc:.0%}")
        print(f"    Time:      {elapsed:.4f}s")
        print(f"    Subsets:   {n_subsets:,}")
        print(f"    Top-5 MI:  {top5}")

    tracker_data = None
    if tracker:
        tracker_data = tracker.to_json()
        if verbose:
            tracker.report()

    return {
        'method': 'mutual_info',
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'n_samples': n_samples,
        'n_subsets': n_subsets,
        'secret': secret,
        'predicted': predicted,
        'correct': correct,
        'best_mi': round(float(best_mi), 6),
        'theoretical_mi': round(theoretical_mi, 6),
        'max_wrong_mi': round(float(max_wrong_mi), 6),
        'mean_wrong_mi': round(float(mean_wrong_mi), 6),
        'mi_gap': round(float(best_mi - max_wrong_mi), 6),
        'test_acc': round(te_acc, 4),
        'elapsed_s': round(elapsed, 6),
        'seed': seed,
        'top5': top5,
        'tracker': tracker_data,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: Mutual Information Estimation for Sparse Parity")
    print("  Blank-slate approach — no neural net, no SGD")
    print("=" * 70)

    all_results = {}
    seeds = [42, 43, 44]

    # -------------------------------------------------------------------
    # 1) n=20, k=3: C(20,3) = 1,140 subsets
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=20, k=3 (C(20,3) = 1,140 subsets)")
    print("=" * 70)

    results_20_3 = []
    for seed in seeds:
        r = run_config(20, 3, n_samples=500, seed=seed,
                       use_tracker=(seed == seeds[0]), verbose=(seed == seeds[0]))
        results_20_3.append(r)
        if seed != seeds[0]:
            status = "OK" if r['correct'] else "FAIL"
            print(f"    seed={seed}: {r['elapsed_s']:.4f}s  {status}  MI={r['best_mi']:.6f}  gap={r['mi_gap']:.6f}")
    all_results['n20_k3'] = results_20_3

    # -------------------------------------------------------------------
    # 2) n=50, k=3: C(50,3) = 19,600 subsets
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=50, k=3 (C(50,3) = 19,600 subsets)")
    print("=" * 70)

    results_50_3 = []
    for seed in seeds:
        r = run_config(50, 3, n_samples=500, seed=seed,
                       use_tracker=False, verbose=(seed == seeds[0]))
        results_50_3.append(r)
        if seed != seeds[0]:
            status = "OK" if r['correct'] else "FAIL"
            print(f"    seed={seed}: {r['elapsed_s']:.4f}s  {status}  MI={r['best_mi']:.6f}  gap={r['mi_gap']:.6f}")
    all_results['n50_k3'] = results_50_3

    # -------------------------------------------------------------------
    # 3) n=20, k=5: C(20,5) = 15,504 subsets
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 3: n=20, k=5 (C(20,5) = 15,504 subsets)")
    print("=" * 70)

    results_20_5 = []
    for seed in seeds:
        r = run_config(20, 5, n_samples=500, seed=seed,
                       use_tracker=False, verbose=(seed == seeds[0]))
        results_20_5.append(r)
        if seed != seeds[0]:
            status = "OK" if r['correct'] else "FAIL"
            print(f"    seed={seed}: {r['elapsed_s']:.4f}s  {status}  MI={r['best_mi']:.6f}  gap={r['mi_gap']:.6f}")
    all_results['n20_k5'] = results_20_5

    # -------------------------------------------------------------------
    # 4) Sample complexity: how few samples do we need for MI?
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SAMPLE COMPLEXITY: n=20, k=3 — minimum samples for MI?")
    print("=" * 70)

    sample_results = []
    for n_samp in [10, 20, 50, 100, 200, 500]:
        correct_count = 0
        total_time = 0
        avg_gap = 0
        for seed in seeds:
            r = run_config(20, 3, n_samples=n_samp, seed=seed,
                           use_tracker=False, verbose=False)
            correct_count += int(r['correct'])
            total_time += r['elapsed_s']
            avg_gap += r['mi_gap']
        avg_time = total_time / len(seeds)
        avg_gap = avg_gap / len(seeds)
        print(f"    n_samples={n_samp:>4}: {correct_count}/{len(seeds)} correct, "
              f"avg MI gap={avg_gap:.4f}, avg {avg_time:.4f}s")
        sample_results.append({
            'n_samples': n_samp,
            'correct': correct_count,
            'total': len(seeds),
            'avg_time_s': round(avg_time, 6),
            'avg_mi_gap': round(avg_gap, 6),
        })
    all_results['sample_complexity'] = sample_results

    # -------------------------------------------------------------------
    # Comparison table: MI vs Fourier vs SGD
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  COMPARISON: MI vs Fourier vs SGD Baselines")
    print("=" * 90)
    header = f"  {'Config':<25} | {'Method':<15} | {'Acc':>5} | {'Time':>10} | {'Notes'}"
    print(header)
    print("  " + "-" * 85)

    for key in ['n20_k3', 'n50_k3', 'n20_k5']:
        runs = all_results[key]
        avg_acc = np.mean([r['test_acc'] for r in runs])
        avg_time = np.mean([r['elapsed_s'] for r in runs])
        n_correct = sum(1 for r in runs if r['correct'])
        avg_mi = np.mean([r['best_mi'] for r in runs])
        avg_gap = np.mean([r['mi_gap'] for r in runs])
        n_sub = runs[0]['n_subsets']
        print(f"  {key:<25} | {'MI':<15} | {avg_acc:>4.0%} | {avg_time:>9.4f}s | "
              f"{n_correct}/{len(runs)} correct, MI={avg_mi:.4f}, gap={avg_gap:.4f}")

    # Fourier baselines from exp_fourier
    print(f"  {'n20_k3':<25} | {'fourier':<15} | {'100%':>5} | {'0.009s':>10} | corr=1.0")
    print(f"  {'n50_k3':<25} | {'fourier':<15} | {'100%':>5} | {'0.13s':>10} | corr=1.0")
    print(f"  {'n20_k5':<25} | {'fourier':<15} | {'100%':>5} | {'1.2s':>10} | corr=1.0")

    # SGD baselines from DISCOVERIES.md
    print(f"  {'n20_k3':<25} | {'SGD':<15} | {'100%':>5} | {'0.12s':>10} | 5 epochs")
    print(f"  {'n50_k3':<25} | {'SGD direct':<15} | {'54%':>5} | {'---':>10} | FAIL")
    print(f"  {'n50_k3':<25} | {'curriculum':<15} | {'>90%':>5} | {'---':>10} | 20 epochs")
    print(f"  {'n20_k5':<25} | {'SGD (n=5000)':<15} | {'100%':>5} | {'---':>10} | 14 epochs")
    print("=" * 90)

    # -------------------------------------------------------------------
    # ARD comparison (first seed tracker data)
    # -------------------------------------------------------------------
    tracker_r = all_results['n20_k3'][0]
    if tracker_r.get('tracker'):
        t = tracker_r['tracker']
        print(f"\n  ARD Report (n=20, k=3, seed=42):")
        print(f"    Total floats accessed: {t['total_floats_accessed']:,}")
        print(f"    Weighted ARD: {t['weighted_ard']:,.0f}")
        print(f"    Reads: {t['reads']:,}, Writes: {t['writes']:,}")
        print(f"    (SGD baseline ARD: 17,976)")
        print(f"    (Fourier ARD: 1,147,375)")

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_mutual_info'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_mutual_info',
            'description': 'Mutual Information estimation over k-subsets for sparse parity',
            'hypothesis': 'MI(y; prod(x_S)) = log(2) for true subset, ~0 for wrong subsets',
            'approach': 'blank_slate — exhaustive MI over all C(n,k) subsets',
            'configs': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")

    return all_results


if __name__ == '__main__':
    main()

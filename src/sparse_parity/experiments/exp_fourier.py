#!/usr/bin/env python3
"""
Experiment: Fourier/Walsh-Hadamard solver for sparse parity.

Blank-slate approach — no neural net, no SGD, no backprop.

Sparse parity with secret S means: label = product(x[i] for i in S).
This is exactly the Walsh-Hadamard coefficient for subset S.
We find it by testing correlations: for each k-subset, compute
mean(y * product(x[:, S], axis=1)). The true subset has correlation ~1.0.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 -m sparse_parity.experiments.exp_fourier
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from itertools import combinations
from math import comb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sparse_parity.tracker import MemTracker


def generate_data(n_bits, k_sparse, n_samples, seed=42):
    """Generate sparse parity data with random secret."""
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    return x, y, secret


def fourier_solve(x, y, n_bits, k_sparse, tracker=None):
    """
    Find the secret subset by exhaustive Walsh-Hadamard correlation.

    For each k-subset S of {0..n-1}:
        correlation(S) = mean(y * product(x[:, S], axis=1))
    The true subset has |correlation| ~= 1.0, all others ~= 0.

    Returns (predicted_secret, correlations_dict, best_corr).
    """
    n_samples = x.shape[0]

    if tracker:
        tracker.write('x', x.size)
        tracker.write('y', n_samples)

    best_corr = 0.0
    best_subset = None
    n_subsets = comb(n_bits, k_sparse)

    for subset in combinations(range(n_bits), k_sparse):
        if tracker:
            tracker.read('x', n_samples * k_sparse)  # read k columns
            tracker.read('y', n_samples)

        # Compute Walsh-Hadamard coefficient for this subset
        parity = np.prod(x[:, list(subset)], axis=1)
        corr = np.abs(np.mean(y * parity))

        if corr > best_corr:
            best_corr = corr
            best_subset = list(subset)

    return best_subset, best_corr, n_subsets


def run_config(n_bits, k_sparse, n_samples, seed=42, use_tracker=False, verbose=True):
    """Run Fourier solver on one config. Returns result dict."""
    n_subsets = comb(n_bits, k_sparse)

    if verbose:
        print(f"  n={n_bits}, k={k_sparse}: {n_subsets:,} subsets, {n_samples} samples")

    x, y, secret = generate_data(n_bits, k_sparse, n_samples, seed=seed)

    tracker = MemTracker() if use_tracker else None

    start = time.time()
    predicted, best_corr, _ = fourier_solve(x, y, n_bits, k_sparse, tracker=tracker)
    elapsed = time.time() - start

    correct = (predicted == secret)

    # Verify: test on fresh data using the PREDICTED secret as the classifier
    rng_te = np.random.RandomState(seed + 1000)
    x_te = rng_te.choice([-1.0, 1.0], size=(500, n_bits))
    # True labels from actual secret
    y_te = np.prod(x_te[:, secret], axis=1)
    # Predictions from our found subset
    y_pred = np.prod(x_te[:, predicted], axis=1)
    te_acc = float(np.mean(y_pred == y_te))

    if verbose:
        status = "CORRECT" if correct else "WRONG"
        print(f"    Secret:    {secret}")
        print(f"    Predicted: {predicted} ({status})")
        print(f"    Correlation: {best_corr:.6f}")
        print(f"    Test acc: {te_acc:.0%}")
        print(f"    Time: {elapsed:.4f}s")
        print(f"    Subsets checked: {n_subsets:,}")

    tracker_data = None
    if tracker:
        tracker_data = tracker.to_json()
        if verbose:
            tracker.report()

    return {
        'method': 'fourier',
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'n_samples': n_samples,
        'n_subsets': n_subsets,
        'secret': secret,
        'predicted': predicted,
        'correct': correct,
        'best_corr': round(float(best_corr), 6),
        'test_acc': round(te_acc, 4),
        'elapsed_s': round(elapsed, 6),
        'seed': seed,
        'tracker': tracker_data,
    }


def main():
    print("=" * 70)
    print("  EXPERIMENT: Fourier/Walsh-Hadamard Solver for Sparse Parity")
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
            print(f"    seed={seed}: {r['elapsed_s']:.4f}s  {status}  corr={r['best_corr']:.4f}")
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
            print(f"    seed={seed}: {r['elapsed_s']:.4f}s  {status}  corr={r['best_corr']:.4f}")
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
            print(f"    seed={seed}: {r['elapsed_s']:.4f}s  {status}  corr={r['best_corr']:.4f}")
    all_results['n20_k5'] = results_20_5

    # -------------------------------------------------------------------
    # 4) Sample complexity: how few samples do we need?
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SAMPLE COMPLEXITY: n=20, k=3 — minimum samples needed?")
    print("=" * 70)

    sample_results = []
    for n_samp in [10, 20, 50, 100, 200, 500]:
        correct_count = 0
        total_time = 0
        for seed in seeds:
            r = run_config(20, 3, n_samples=n_samp, seed=seed,
                           use_tracker=False, verbose=False)
            correct_count += int(r['correct'])
            total_time += r['elapsed_s']
        avg_time = total_time / len(seeds)
        print(f"    n_samples={n_samp:>4}: {correct_count}/{len(seeds)} correct, avg {avg_time:.4f}s")
        sample_results.append({
            'n_samples': n_samp,
            'correct': correct_count,
            'total': len(seeds),
            'avg_time_s': round(avg_time, 6),
        })
    all_results['sample_complexity'] = sample_results

    # -------------------------------------------------------------------
    # 5) Scaling: larger n and k
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SCALING: How far can Fourier go?")
    print("=" * 70)

    scaling_results = []
    for n_bits, k_sparse in [(100, 3), (200, 3), (20, 7)]:
        n_sub = comb(n_bits, k_sparse)
        print(f"\n  n={n_bits}, k={k_sparse}: C({n_bits},{k_sparse}) = {n_sub:,} subsets")
        if n_sub > 5_000_000:
            print(f"    SKIPPED: too many subsets ({n_sub:,})")
            scaling_results.append({
                'n_bits': n_bits, 'k_sparse': k_sparse,
                'n_subsets': n_sub, 'skipped': True,
            })
            continue
        r = run_config(n_bits, k_sparse, n_samples=500, seed=42,
                       use_tracker=False, verbose=True)
        scaling_results.append(r)
    all_results['scaling'] = scaling_results

    # -------------------------------------------------------------------
    # Comparison table
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  COMPARISON: Fourier vs SGD Baselines (from DISCOVERIES.md)")
    print("=" * 90)
    header = f"  {'Config':<25} | {'Method':<15} | {'Acc':>5} | {'Time':>10} | {'Notes'}"
    print(header)
    print("  " + "-" * 85)

    # Fourier results
    for key in ['n20_k3', 'n50_k3', 'n20_k5']:
        runs = all_results[key]
        avg_acc = np.mean([r['test_acc'] for r in runs])
        avg_time = np.mean([r['elapsed_s'] for r in runs])
        n_correct = sum(1 for r in runs if r['correct'])
        n_sub = runs[0]['n_subsets']
        print(f"  {key:<25} | {'fourier':<15} | {avg_acc:>4.0%} | {avg_time:>9.4f}s | {n_correct}/{len(runs)} correct, {n_sub:,} subsets")

    # SGD baselines
    print(f"  {'n20_k3':<25} | {'SGD (fast.py)':<15} | {'100%':>5} | {'0.12s':>10} | baseline")
    print(f"  {'n50_k3':<25} | {'curriculum':<15} | {'>90%':>5} | {'—':>10} | 20 epochs")
    print(f"  {'n50_k3':<25} | {'SGD direct':<15} | {'54%':>5} | {'—':>10} | FAIL")
    print(f"  {'n20_k5':<25} | {'SGD (n=5000)':<15} | {'100%':>5} | {'—':>10} | 14 epochs")
    print(f"  {'n20_k5':<25} | {'sign SGD':<15} | {'100%':>5} | {'—':>10} | 7 epochs")
    print("=" * 90)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_fourier'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_fourier',
            'description': 'Fourier/Walsh-Hadamard solver — exhaustive correlation search',
            'hypothesis': 'Exhaustive Walsh-Hadamard correlation finds secret in O(C(n,k)) time with O(n) samples',
            'configs': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")

    return all_results


if __name__ == '__main__':
    main()

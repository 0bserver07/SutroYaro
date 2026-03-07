#!/usr/bin/env python3
"""
Experiment: Minimum Description Length (MDL) solver for sparse parity.

Blank-slate approach -- no neural net, no SGD, no backprop.

Idea: The best compressor of the label sequence is the one that knows
the secret bits. For each candidate k-subset S, compute the parity of
the inputs restricted to S and compare with y. The description length
of the residual (how many labels disagree) is:

    DL(S) = n_samples * H(residual_rate)

where H is binary entropy. For the true subset, residual_rate = 0 and
DL = 0. For wrong subsets, residual_rate ~= 0.5 and DL ~= n_samples bits.

This is more general than Fourier: it works for any deterministic
labeling function, not just parity.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_mdl.py
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


# =============================================================================
# UTILITIES
# =============================================================================

def binary_entropy(p):
    """Binary entropy H(p) in bits. H(0)=H(1)=0."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def generate_data(n_bits, k_sparse, n_samples, seed=42, noise_rate=0.0):
    """Generate sparse parity data with random secret, optional label noise."""
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    if noise_rate > 0.0:
        flip_mask = rng.random(n_samples) < noise_rate
        y[flip_mask] *= -1.0
    return x, y, secret


# =============================================================================
# MDL SOLVER
# =============================================================================

def mdl_solve(x, y, n_bits, k_sparse, tracker=None):
    """
    Find the secret subset by minimum description length.

    For each k-subset S of {0..n-1}:
        parity_S = product(x[:, S], axis=1)
        residual_rate = mean(parity_S != y)
        description_length = n_samples * H(residual_rate)

    The true subset has DL = 0 (all labels match). Wrong subsets have
    DL ~= n_samples bits.

    Returns (predicted_secret, best_dl, n_subsets, all_description_lengths).
    """
    n_samples = x.shape[0]

    if tracker:
        tracker.write('x', x.size)
        tracker.write('y', n_samples)

    best_dl = float('inf')
    best_subset = None
    n_subsets = comb(n_bits, k_sparse)
    description_lengths = []

    for subset in combinations(range(n_bits), k_sparse):
        if tracker:
            tracker.read('x', n_samples * k_sparse)
            tracker.read('y', n_samples)

        parity = np.prod(x[:, list(subset)], axis=1)
        residual_rate = float(np.mean(parity != y))
        dl = n_samples * binary_entropy(residual_rate)

        description_lengths.append((list(subset), dl, residual_rate))

        if dl < best_dl:
            best_dl = dl
            best_subset = list(subset)

    return best_subset, best_dl, n_subsets, description_lengths


# =============================================================================
# FOURIER SOLVER (for comparison)
# =============================================================================

def fourier_solve(x, y, n_bits, k_sparse):
    """Walsh-Hadamard correlation solver for comparison."""
    best_corr = 0.0
    best_subset = None

    for subset in combinations(range(n_bits), k_sparse):
        parity = np.prod(x[:, list(subset)], axis=1)
        corr = np.abs(np.mean(y * parity))
        if corr > best_corr:
            best_corr = corr
            best_subset = list(subset)

    return best_subset, best_corr


# =============================================================================
# SINGLE CONFIG RUNNER
# =============================================================================

def run_config(n_bits, k_sparse, n_samples, seed=42, noise_rate=0.0,
               use_tracker=False, verbose=True):
    """Run MDL solver on one config. Returns result dict."""
    n_subsets = comb(n_bits, k_sparse)

    if verbose:
        noise_str = f", noise={noise_rate:.0%}" if noise_rate > 0 else ""
        print(f"  n={n_bits}, k={k_sparse}: {n_subsets:,} subsets, "
              f"{n_samples} samples{noise_str}")

    x, y, secret = generate_data(n_bits, k_sparse, n_samples, seed=seed,
                                  noise_rate=noise_rate)

    tracker = MemTracker() if use_tracker else None

    # MDL solve
    start = time.time()
    predicted, best_dl, _, dl_list = mdl_solve(
        x, y, n_bits, k_sparse, tracker=tracker
    )
    mdl_elapsed = time.time() - start

    mdl_correct = (predicted == secret)

    # Fourier comparison
    start_f = time.time()
    fourier_pred, fourier_corr = fourier_solve(x, y, n_bits, k_sparse)
    fourier_elapsed = time.time() - start_f
    fourier_correct = (fourier_pred == secret)

    # Test accuracy on fresh data (no noise)
    rng_te = np.random.RandomState(seed + 1000)
    x_te = rng_te.choice([-1.0, 1.0], size=(500, n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    y_pred_mdl = np.prod(x_te[:, predicted], axis=1)
    te_acc_mdl = float(np.mean(y_pred_mdl == y_te))
    y_pred_four = np.prod(x_te[:, fourier_pred], axis=1)
    te_acc_four = float(np.mean(y_pred_four == y_te))

    # Stats on description lengths
    dl_values = [dl for _, dl, _ in dl_list]
    residual_rates = [rr for _, _, rr in dl_list]

    # The true subset DL vs average wrong subset DL
    true_dl = None
    for s, dl, rr in dl_list:
        if s == secret:
            true_dl = dl
            break

    if verbose:
        status_mdl = "CORRECT" if mdl_correct else "WRONG"
        status_four = "CORRECT" if fourier_correct else "WRONG"
        print(f"    Secret:         {secret}")
        print(f"    MDL predicted:  {predicted} ({status_mdl})")
        print(f"    MDL best DL:    {best_dl:.4f} bits")
        print(f"    True subset DL: {true_dl:.4f} bits" if true_dl is not None else "")
        print(f"    Avg wrong DL:   {np.mean([dl for s, dl, _ in dl_list if s != secret]):.2f} bits")
        print(f"    MDL time:       {mdl_elapsed:.4f}s")
        print(f"    Fourier pred:   {fourier_pred} ({status_four}), corr={fourier_corr:.6f}")
        print(f"    Fourier time:   {fourier_elapsed:.4f}s")
        print(f"    Test acc MDL:   {te_acc_mdl:.0%}")
        print(f"    Test acc Four:  {te_acc_four:.0%}")

    tracker_data = None
    if tracker:
        tracker_data = tracker.to_json()
        if verbose:
            tracker.report()

    return {
        'method': 'mdl',
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'n_samples': n_samples,
        'n_subsets': n_subsets,
        'noise_rate': noise_rate,
        'secret': secret,
        'mdl_predicted': predicted,
        'mdl_correct': mdl_correct,
        'mdl_best_dl': round(float(best_dl), 6),
        'mdl_true_dl': round(float(true_dl), 6) if true_dl is not None else None,
        'mdl_elapsed_s': round(mdl_elapsed, 6),
        'mdl_test_acc': round(te_acc_mdl, 4),
        'fourier_predicted': fourier_pred,
        'fourier_correct': fourier_correct,
        'fourier_best_corr': round(float(fourier_corr), 6),
        'fourier_elapsed_s': round(fourier_elapsed, 6),
        'fourier_test_acc': round(te_acc_four, 4),
        'dl_stats': {
            'mean': round(float(np.mean(dl_values)), 4),
            'std': round(float(np.std(dl_values)), 4),
            'min': round(float(np.min(dl_values)), 6),
            'max': round(float(np.max(dl_values)), 4),
            'median': round(float(np.median(dl_values)), 4),
        },
        'residual_rate_stats': {
            'mean': round(float(np.mean(residual_rates)), 6),
            'std': round(float(np.std(residual_rates)), 6),
            'min': round(float(np.min(residual_rates)), 6),
            'max': round(float(np.max(residual_rates)), 6),
        },
        'seed': seed,
        'tracker': tracker_data,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: Minimum Description Length (MDL) for Sparse Parity")
    print("  Blank-slate approach -- no neural net, no SGD")
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
                       use_tracker=(seed == seeds[0]),
                       verbose=(seed == seeds[0]))
        results_20_3.append(r)
        if seed != seeds[0]:
            status = "OK" if r['mdl_correct'] else "FAIL"
            print(f"    seed={seed}: MDL {r['mdl_elapsed_s']:.4f}s {status} "
                  f"| Fourier {r['fourier_elapsed_s']:.4f}s "
                  f"{'OK' if r['fourier_correct'] else 'FAIL'}")
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
            status = "OK" if r['mdl_correct'] else "FAIL"
            print(f"    seed={seed}: MDL {r['mdl_elapsed_s']:.4f}s {status} "
                  f"| Fourier {r['fourier_elapsed_s']:.4f}s "
                  f"{'OK' if r['fourier_correct'] else 'FAIL'}")
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
            status = "OK" if r['mdl_correct'] else "FAIL"
            print(f"    seed={seed}: MDL {r['mdl_elapsed_s']:.4f}s {status} "
                  f"| Fourier {r['fourier_elapsed_s']:.4f}s "
                  f"{'OK' if r['fourier_correct'] else 'FAIL'}")
    all_results['n20_k5'] = results_20_5

    # -------------------------------------------------------------------
    # 4) Noise robustness: 5% label noise
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  NOISE ROBUSTNESS: 5% label noise")
    print("=" * 70)

    noise_results = []
    configs = [
        (20, 3, 500, 'n20_k3_noise'),
        (50, 3, 500, 'n50_k3_noise'),
        (20, 5, 500, 'n20_k5_noise'),
    ]
    for n_bits, k_sparse, n_samp, label in configs:
        print(f"\n  --- {label} ---")
        cfg_results = []
        for seed in seeds:
            r = run_config(n_bits, k_sparse, n_samples=n_samp, seed=seed,
                           noise_rate=0.05, use_tracker=False,
                           verbose=(seed == seeds[0]))
            cfg_results.append(r)
            if seed != seeds[0]:
                status = "OK" if r['mdl_correct'] else "FAIL"
                print(f"    seed={seed}: MDL {r['mdl_elapsed_s']:.4f}s {status} "
                      f"DL={r['mdl_best_dl']:.2f} "
                      f"| Fourier {'OK' if r['fourier_correct'] else 'FAIL'} "
                      f"corr={r['fourier_best_corr']:.4f}")
        noise_results.append({'config': label, 'runs': cfg_results})
    all_results['noise_robustness'] = noise_results

    # -------------------------------------------------------------------
    # Summary tables
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SUMMARY: MDL vs Fourier (no noise)")
    print("=" * 90)
    header = (f"  {'Config':<15} | {'MDL correct':>11} | {'MDL avg time':>12} | "
              f"{'Fourier correct':>15} | {'Fourier avg time':>16}")
    print(header)
    print("  " + "-" * 80)

    for key in ['n20_k3', 'n50_k3', 'n20_k5']:
        runs = all_results[key]
        mdl_corr = sum(1 for r in runs if r['mdl_correct'])
        mdl_time = np.mean([r['mdl_elapsed_s'] for r in runs])
        four_corr = sum(1 for r in runs if r['fourier_correct'])
        four_time = np.mean([r['fourier_elapsed_s'] for r in runs])
        print(f"  {key:<15} | {mdl_corr}/{len(runs):>9} | {mdl_time:>11.4f}s | "
              f"{four_corr}/{len(runs):>13} | {four_time:>15.4f}s")

    print("\n" + "=" * 90)
    print("  SUMMARY: Noise robustness (5% label noise)")
    print("=" * 90)
    header = (f"  {'Config':<20} | {'MDL correct':>11} | {'MDL best DL':>11} | "
              f"{'Fourier correct':>15} | {'Fourier best corr':>18}")
    print(header)
    print("  " + "-" * 85)

    for nr in all_results['noise_robustness']:
        runs = nr['runs']
        label = nr['config']
        mdl_corr = sum(1 for r in runs if r['mdl_correct'])
        mdl_dl = np.mean([r['mdl_best_dl'] for r in runs])
        four_corr = sum(1 for r in runs if r['fourier_correct'])
        four_best = np.mean([r['fourier_best_corr'] for r in runs])
        print(f"  {label:<20} | {mdl_corr}/{len(runs):>9} | {mdl_dl:>10.2f} | "
              f"{four_corr}/{len(runs):>13} | {four_best:>17.4f}")

    print("=" * 90)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_mdl'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_mdl',
            'description': 'Minimum Description Length solver for sparse parity',
            'hypothesis': 'The best compressor of the label sequence knows the secret bits; '
                          'MDL is more general than Fourier and works under label noise.',
            'configs': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

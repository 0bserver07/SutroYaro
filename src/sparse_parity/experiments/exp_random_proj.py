#!/usr/bin/env python3
"""
Experiment: Random Projections + Correlation for Sparse Parity

Approach #5: Monte Carlo version of the Fourier solver. Instead of testing
ALL C(n,k) subsets exhaustively, sample random k-subsets and compute
Walsh-Hadamard correlation. Stop early when |corr| > 0.9.

Expected C(n,k) evaluations on average (geometric distribution), but early
stopping means we don't always scan the full space.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_random_proj.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from math import comb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sparse_parity.tracker import MemTracker


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n_bits, k_sparse, n_samples, seed=42):
    """Generate sparse parity data with random secret."""
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    return x, y, secret


# =============================================================================
# RANDOM PROJECTIONS + CORRELATION (Monte Carlo Walsh-Hadamard)
# =============================================================================

def random_proj_solve(x, y, n_bits, k_sparse, max_tries=500000,
                      corr_threshold=0.9, seed=42, tracker=None):
    """
    Monte Carlo Walsh-Hadamard: sample random k-subsets, compute correlation.

    For each sampled subset S:
        corr = mean(y * product(x[:, S], axis=1))
    If |corr| > corr_threshold, declare found and stop.

    Returns (predicted_subset, n_tried, elapsed_s, best_corr).
    """
    rng = np.random.RandomState(seed)
    n_samples = x.shape[0]

    if tracker:
        tracker.write('x', x.size)
        tracker.write('y', n_samples)

    best_corr = 0.0
    best_subset = None
    tried = set()

    start = time.time()

    for t in range(1, max_tries + 1):
        subset = tuple(sorted(rng.choice(n_bits, k_sparse, replace=False).tolist()))

        # Skip duplicates
        if subset in tried:
            continue
        tried.add(subset)

        if tracker:
            tracker.read('x', n_samples * k_sparse)
            tracker.read('y', n_samples)

        parity = np.prod(x[:, list(subset)], axis=1)
        corr = np.abs(np.mean(y * parity))

        if corr > best_corr:
            best_corr = corr
            best_subset = list(subset)

        if corr > corr_threshold:
            elapsed = time.time() - start
            return best_subset, len(tried), elapsed, float(best_corr)

    elapsed = time.time() - start
    return best_subset, len(tried), elapsed, float(best_corr)


# =============================================================================
# EXHAUSTIVE FOURIER (for comparison)
# =============================================================================

def fourier_solve(x, y, n_bits, k_sparse):
    """Exhaustive Walsh-Hadamard — tests ALL C(n,k) subsets."""
    from itertools import combinations

    best_corr = 0.0
    best_subset = None
    n_subsets = comb(n_bits, k_sparse)

    start = time.time()
    for subset in combinations(range(n_bits), k_sparse):
        parity = np.prod(x[:, list(subset)], axis=1)
        corr = np.abs(np.mean(y * parity))
        if corr > best_corr:
            best_corr = corr
            best_subset = list(subset)
    elapsed = time.time() - start

    return best_subset, n_subsets, elapsed, float(best_corr)


# =============================================================================
# RUN ONE CONFIG
# =============================================================================

def run_config(n_bits, k_sparse, n_samples, seeds, max_tries=500000,
               corr_threshold=0.9, run_fourier=True, verbose=True):
    """Run random projection solver across seeds. Optionally compare to Fourier."""
    c_n_k = comb(n_bits, k_sparse)

    if verbose:
        print(f"\n  Config: n={n_bits}, k={k_sparse}, C(n,k)={c_n_k:,}")
        print(f"  Samples: {n_samples}, Max tries: {max_tries}, Threshold: {corr_threshold}")

    rp_results = []
    fourier_results = []

    for seed in seeds:
        x, y, secret = generate_data(n_bits, k_sparse, n_samples, seed=seed)

        # --- Random Projections ---
        use_tracker = (seed == seeds[0])
        tracker = MemTracker() if use_tracker else None

        pred, n_tried, elapsed, best_corr = random_proj_solve(
            x, y, n_bits, k_sparse,
            max_tries=max_tries, corr_threshold=corr_threshold,
            seed=seed + 300, tracker=tracker
        )

        solved = best_corr > corr_threshold
        correct = (sorted(pred) == secret) if solved else False

        # Test accuracy on fresh data
        rng_te = np.random.RandomState(seed + 1000)
        x_te = rng_te.choice([-1.0, 1.0], size=(500, n_bits))
        y_te = np.prod(x_te[:, secret], axis=1)
        y_pred = np.prod(x_te[:, pred], axis=1) if pred else np.zeros(500)
        te_acc = float(np.mean(y_pred == y_te))

        tracker_data = tracker.to_json() if tracker else None
        if tracker and verbose:
            tracker.report()

        rp_results.append({
            'seed': seed,
            'predicted': pred,
            'secret': secret,
            'correct': correct,
            'solved': solved,
            'n_tried': n_tried,
            'best_corr': round(best_corr, 6),
            'test_acc': round(te_acc, 4),
            'elapsed_s': round(elapsed, 6),
            'tracker': tracker_data,
        })

        if verbose:
            status = "SOLVED" if solved else "FAILED"
            pct = n_tried / c_n_k * 100
            print(f"    [RandProj] seed={seed}: {status} in {n_tried:,} tries "
                  f"({pct:.1f}% of C(n,k)) corr={best_corr:.4f} "
                  f"time={elapsed:.4f}s acc={te_acc:.0%}")

        # --- Fourier (exhaustive) for comparison ---
        if run_fourier:
            f_pred, f_n, f_elapsed, f_corr = fourier_solve(
                x, y, n_bits, k_sparse
            )
            f_correct = (sorted(f_pred) == secret)

            fourier_results.append({
                'seed': seed,
                'predicted': f_pred,
                'correct': f_correct,
                'n_subsets': f_n,
                'best_corr': round(f_corr, 6),
                'elapsed_s': round(f_elapsed, 6),
            })

            if verbose:
                print(f"    [Fourier]  seed={seed}: {f_n:,} subsets "
                      f"corr={f_corr:.4f} time={f_elapsed:.4f}s")

    return {
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'c_n_k': c_n_k,
        'n_samples': n_samples,
        'random_proj': rp_results,
        'fourier': fourier_results if run_fourier else None,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: Random Projections + Correlation for Sparse Parity")
    print("  Monte Carlo Walsh-Hadamard with early stopping")
    print("=" * 70)

    seeds = [42, 43, 44, 45, 46]
    all_results = {}

    # -------------------------------------------------------------------
    # Config 1: n=20, k=3 — C(20,3) = 1,140
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=20, k=3  [C(20,3) = 1,140]")
    print("=" * 70)
    all_results['n20_k3'] = run_config(
        n_bits=20, k_sparse=3, n_samples=500, seeds=seeds,
        max_tries=50000, run_fourier=True
    )

    # -------------------------------------------------------------------
    # Config 2: n=50, k=3 — C(50,3) = 19,600
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=50, k=3  [C(50,3) = 19,600]")
    print("=" * 70)
    all_results['n50_k3'] = run_config(
        n_bits=50, k_sparse=3, n_samples=500, seeds=seeds,
        max_tries=200000, run_fourier=True
    )

    # -------------------------------------------------------------------
    # Config 3: n=20, k=5 — C(20,5) = 15,504
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 3: n=20, k=5  [C(20,5) = 15,504]")
    print("=" * 70)
    all_results['n20_k5'] = run_config(
        n_bits=20, k_sparse=5, n_samples=500, seeds=seeds,
        max_tries=200000, run_fourier=True
    )

    # -------------------------------------------------------------------
    # Scaling: n=100, k=3 — C(100,3) = 161,700
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SCALING 1: n=100, k=3  [C(100,3) = 161,700]")
    print("=" * 70)
    all_results['n100_k3'] = run_config(
        n_bits=100, k_sparse=3, n_samples=500, seeds=seeds,
        max_tries=500000, run_fourier=False  # Fourier too slow at this scale
    )

    # -------------------------------------------------------------------
    # Scaling: n=200, k=3 — C(200,3) = 1,313,400
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SCALING 2: n=200, k=3  [C(200,3) = 1,313,400]")
    print("=" * 70)
    all_results['n200_k3'] = run_config(
        n_bits=200, k_sparse=3, n_samples=500, seeds=seeds,
        max_tries=5000000, run_fourier=False  # Fourier too slow
    )

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("  SUMMARY TABLE")
    print("=" * 100)
    header = (f"  {'Config':<12} | {'C(n,k)':>10} | "
              f"{'RP tries':>10} | {'RP %C(n,k)':>10} | {'RP time':>9} | "
              f"{'Fourier time':>12} | {'Speedup':>8} | {'RP solved':>9}")
    print(header)
    print("  " + "-" * 98)

    for key, res in all_results.items():
        c_n_k = res['c_n_k']

        rp_solved = sum(1 for r in res['random_proj'] if r['solved'])
        rp_solved_runs = [r for r in res['random_proj'] if r['solved']]

        if rp_solved > 0:
            rp_tries_avg = np.mean([r['n_tried'] for r in rp_solved_runs])
            rp_time_avg = np.mean([r['elapsed_s'] for r in rp_solved_runs])
            rp_pct = rp_tries_avg / c_n_k * 100
            rp_tries_str = f"{rp_tries_avg:,.0f}"
            rp_pct_str = f"{rp_pct:.1f}%"
            rp_time_str = f"{rp_time_avg:.4f}s"
        else:
            rp_tries_str = "FAIL"
            rp_pct_str = "---"
            rp_time_str = "---"

        if res['fourier'] and len(res['fourier']) > 0:
            f_time_avg = np.mean([r['elapsed_s'] for r in res['fourier']])
            f_time_str = f"{f_time_avg:.4f}s"
            if rp_solved > 0:
                speedup = f_time_avg / rp_time_avg
                speedup_str = f"{speedup:.1f}x"
            else:
                speedup_str = "---"
        else:
            f_time_str = "N/A"
            speedup_str = "---"

        print(f"  {key:<12} | {c_n_k:>10,} | "
              f"{rp_tries_str:>10} | {rp_pct_str:>10} | {rp_time_str:>9} | "
              f"{f_time_str:>12} | {speedup_str:>8} | {rp_solved}/{len(res['random_proj']):>8}")

    # -------------------------------------------------------------------
    # Variance analysis
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  VARIANCE ANALYSIS (across 5 seeds)")
    print("=" * 100)
    for key, res in all_results.items():
        solved_runs = [r for r in res['random_proj'] if r['solved']]
        if len(solved_runs) >= 2:
            tries = [r['n_tried'] for r in solved_runs]
            times = [r['elapsed_s'] for r in solved_runs]
            print(f"  {key:<12}: tries mean={np.mean(tries):.0f} std={np.std(tries):.0f} "
                  f"min={min(tries)} max={max(tries)} | "
                  f"time mean={np.mean(times):.4f}s std={np.std(times):.4f}s")

    # -------------------------------------------------------------------
    # Comparison with other methods
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  COMPARISON: Random Projections vs Fourier vs SGD")
    print("=" * 100)
    print(f"  {'Config':<15} | {'Method':<20} | {'Acc':>5} | {'Time':>10} | {'Subsets tested':>15}")
    print("  " + "-" * 75)

    for key in ['n20_k3', 'n50_k3', 'n20_k5']:
        res = all_results[key]
        c_n_k = res['c_n_k']

        # Random projections
        rp_solved_runs = [r for r in res['random_proj'] if r['solved']]
        if rp_solved_runs:
            rp_acc = np.mean([r['test_acc'] for r in rp_solved_runs])
            rp_time = np.mean([r['elapsed_s'] for r in rp_solved_runs])
            rp_tries = np.mean([r['n_tried'] for r in rp_solved_runs])
            print(f"  {key:<15} | {'Random Proj':<20} | {rp_acc:>4.0%} | "
                  f"{rp_time:>9.4f}s | {rp_tries:>14,.0f}")

        # Fourier
        if res['fourier']:
            f_time = np.mean([r['elapsed_s'] for r in res['fourier']])
            print(f"  {'':<15} | {'Fourier (exhaust.)':<20} | {'100%':>5} | "
                  f"{f_time:>9.4f}s | {c_n_k:>14,}")

        # SGD baselines
        if key == 'n20_k3':
            print(f"  {'':<15} | {'SGD (fast.py)':<20} | {'100%':>5} | {'0.12s':>10} | {'N/A':>15}")
        elif key == 'n50_k3':
            print(f"  {'':<15} | {'SGD direct':<20} | {'54%':>5} | {'---':>10} | {'N/A':>15}")
        elif key == 'n20_k5':
            print(f"  {'':<15} | {'SGD (n=5000)':<20} | {'100%':>5} | {'---':>10} | {'N/A':>15}")

    print("=" * 100)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_random_proj'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_random_proj',
            'description': 'Random Projections + Correlation — Monte Carlo Walsh-Hadamard with early stopping',
            'hypothesis': 'Random sampling of k-subsets with correlation test finds secret with early stopping, often faster than exhaustive Fourier',
            'approach': 'blank_slate — no neural net, no SGD, no gradients',
            'configs': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

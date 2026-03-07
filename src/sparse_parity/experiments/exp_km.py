#!/usr/bin/env python3
"""
Experiment exp_km: Kushilevitz-Mansour Algorithm for Sparse Parity

Hypothesis: The KM algorithm identifies the secret bits by estimating per-bit
influence (flip bit i, measure label change rate). Bits in the secret have
influence = 1.0, others have influence = 0.0. This prunes the search from
C(n,k) to C(k',k) where k' is the number of high-influence bits (ideally k'=k).
For sparse parity with k=3, this should need ~O(n) influence queries + O(1)
verification, far fewer total samples than brute-force Fourier.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_km.py
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
# DATA GENERATION
# =============================================================================

def generate_data(n_bits, k_sparse, n_samples, seed=42):
    """Generate sparse parity data with random secret. Returns x, y, secret."""
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    return x, y, secret


def oracle_query(secret, x_single):
    """Oracle: returns the parity label for a single input vector."""
    return np.prod(x_single[secret])


# =============================================================================
# KM ALGORITHM: INFLUENCE ESTIMATION
# =============================================================================

def estimate_influence(n_bits, k_sparse, n_influence_samples, seed=42):
    """
    Estimate the influence of each bit using paired queries.

    For each bit i, we generate random inputs x, compute f(x) and f(x^i)
    (where x^i flips bit i), and measure how often the label changes.

    Influence_i = Pr[f(x) != f(x^i)] over random x.

    For k-parity with secret S:
      - If i in S: flipping i always flips the product, so influence = 1.0
      - If i not in S: flipping i never changes the product, so influence = 0.0

    Returns: influences array, total queries used, secret (for verification).
    """
    rng = np.random.RandomState(seed)

    # Generate the secret (same RNG protocol as generate_data)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())

    # For influence estimation, we generate fresh random inputs
    rng_inf = np.random.RandomState(seed + 500)

    influences = np.zeros(n_bits)
    total_queries = 0

    for i in range(n_bits):
        # Generate n_influence_samples random inputs
        x_batch = rng_inf.choice([-1.0, 1.0], size=(n_influence_samples, n_bits))

        # Compute f(x) for each
        y_orig = np.prod(x_batch[:, secret], axis=1)
        total_queries += n_influence_samples

        # Flip bit i
        x_flipped = x_batch.copy()
        x_flipped[:, i] *= -1

        # Compute f(x^i) for each
        y_flipped = np.prod(x_flipped[:, secret], axis=1)
        total_queries += n_influence_samples

        # Influence = fraction of times label changed
        influences[i] = np.mean(y_orig != y_flipped)

    return influences, total_queries, secret


# =============================================================================
# KM ALGORITHM: FULL PIPELINE
# =============================================================================

def km_solve(n_bits, k_sparse, n_influence_samples=200, influence_threshold=0.5,
             n_verify_samples=100, seed=42, tracker=None):
    """
    Kushilevitz-Mansour algorithm for sparse parity:

    1. Estimate influence of each bit (O(n) paired queries)
    2. Select bits with influence > threshold as candidates
    3. If |candidates| == k, we're done -- verify
    4. If |candidates| > k, enumerate k-subsets of candidates and verify
    5. Verification: check if product(x[:, subset]) == y on fresh samples

    Returns dict with predicted secret, timing, query counts, etc.
    """
    start = time.time()

    if tracker:
        tracker.write('config', 3)  # n_bits, k_sparse, threshold

    # Step 1: Estimate influences
    t_inf_start = time.time()
    influences, inf_queries, secret = estimate_influence(
        n_bits, k_sparse, n_influence_samples, seed=seed
    )
    t_inf = time.time() - t_inf_start

    if tracker:
        tracker.write('influences', n_bits)
        tracker.read('influences', n_bits)

    # Step 2: Select high-influence bits
    candidates = [i for i in range(n_bits) if influences[i] > influence_threshold]
    n_candidates = len(candidates)

    if tracker:
        tracker.write('candidates', n_candidates)

    # Step 3+4: Verify among candidates
    t_verify_start = time.time()
    verify_queries = 0
    predicted = None
    n_subsets_checked = 0

    if n_candidates == k_sparse:
        # Perfect pruning: only one subset to check
        rng_v = np.random.RandomState(seed + 1000)
        x_v = rng_v.choice([-1.0, 1.0], size=(n_verify_samples, n_bits))
        y_v = np.prod(x_v[:, secret], axis=1)
        verify_queries += n_verify_samples

        pred = np.prod(x_v[:, candidates], axis=1)
        verify_queries += n_verify_samples
        n_subsets_checked = 1

        if np.all(pred == y_v):
            predicted = sorted(candidates)

        if tracker:
            tracker.read('candidates', n_candidates)
            tracker.write('x_verify', n_verify_samples * n_bits)
            tracker.write('y_verify', n_verify_samples)
            tracker.read('x_verify', n_verify_samples * k_sparse)
            tracker.read('y_verify', n_verify_samples)

    elif n_candidates >= k_sparse:
        # More candidates than needed: enumerate k-subsets of candidates
        rng_v = np.random.RandomState(seed + 1000)
        x_v = rng_v.choice([-1.0, 1.0], size=(n_verify_samples, n_bits))
        y_v = np.prod(x_v[:, secret], axis=1)
        verify_queries += n_verify_samples

        if tracker:
            tracker.write('x_verify', n_verify_samples * n_bits)
            tracker.write('y_verify', n_verify_samples)

        n_candidate_subsets = comb(n_candidates, k_sparse)

        for subset in combinations(candidates, k_sparse):
            pred = np.prod(x_v[:, list(subset)], axis=1)
            verify_queries += n_verify_samples
            n_subsets_checked += 1

            if tracker:
                tracker.read('x_verify', n_verify_samples * k_sparse)
                tracker.read('y_verify', n_verify_samples)

            if np.all(pred == y_v):
                predicted = sorted(list(subset))
                break
    else:
        # Too few candidates (shouldn't happen with enough samples)
        n_candidate_subsets = 0

    t_verify = time.time() - t_verify_start
    total_time = time.time() - start

    total_queries = inf_queries + verify_queries

    # Test accuracy on fresh data
    rng_te = np.random.RandomState(seed + 2000)
    x_te = rng_te.choice([-1.0, 1.0], size=(500, n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    if predicted is not None:
        y_pred = np.prod(x_te[:, predicted], axis=1)
        test_acc = float(np.mean(y_pred == y_te))
    else:
        test_acc = 0.0

    correct = (predicted == secret) if predicted is not None else False

    return {
        'method': 'km',
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'seed': seed,
        'secret': secret,
        'predicted': predicted,
        'correct': correct,
        'test_acc': test_acc,
        'n_influence_samples': n_influence_samples,
        'n_verify_samples': n_verify_samples,
        'influence_threshold': influence_threshold,
        'influences': {i: round(float(influences[i]), 4) for i in range(n_bits)},
        'candidates': candidates,
        'n_candidates': n_candidates,
        'n_subsets_checked': n_subsets_checked,
        'c_n_k': comb(n_bits, k_sparse),
        'c_cand_k': comb(n_candidates, k_sparse) if n_candidates >= k_sparse else 0,
        'inf_queries': inf_queries,
        'verify_queries': verify_queries,
        'total_queries': total_queries,
        'time_influence_s': round(t_inf, 6),
        'time_verify_s': round(t_verify, 6),
        'time_total_s': round(total_time, 6),
        'tracker': tracker.to_json() if tracker else None,
    }


# =============================================================================
# SAMPLE COMPLEXITY: how few influence samples are needed?
# =============================================================================

def sample_complexity_sweep(n_bits, k_sparse, seed=42, verbose=True):
    """Test how few influence samples are needed for correct identification."""
    results = []
    sample_counts = [5, 10, 20, 30, 50, 100, 200, 500]

    if verbose:
        print(f"\n  Sample complexity sweep: n={n_bits}, k={k_sparse}")
        print(f"  {'Inf samples':>12} | {'Candidates':>10} | {'Correct':>8} | "
              f"{'Total queries':>14} | {'Time':>8}")
        print(f"  {'─'*12} | {'─'*10} | {'─'*8} | {'─'*14} | {'─'*8}")

    for n_inf in sample_counts:
        r = km_solve(n_bits, k_sparse, n_influence_samples=n_inf,
                     n_verify_samples=50, seed=seed)
        results.append({
            'n_influence_samples': n_inf,
            'n_candidates': r['n_candidates'],
            'correct': r['correct'],
            'total_queries': r['total_queries'],
            'time_s': r['time_total_s'],
        })
        if verbose:
            status = "YES" if r['correct'] else "NO"
            print(f"  {n_inf:>12} | {r['n_candidates']:>10} | {status:>8} | "
                  f"{r['total_queries']:>14,} | {r['time_total_s']:>7.4f}s")

    return results


# =============================================================================
# MAIN
# =============================================================================

def run_config(n_bits, k_sparse, n_influence_samples, seeds, verbose=True):
    """Run KM algorithm on one config across multiple seeds."""
    c_n_k = comb(n_bits, k_sparse)
    if verbose:
        print(f"\n  Config: n={n_bits}, k={k_sparse}, C(n,k)={c_n_k:,}")
        print(f"  Influence samples per bit: {n_influence_samples}")

    results = []
    for seed in seeds:
        use_tracker = (seed == seeds[0])
        tracker = MemTracker() if use_tracker else None

        r = km_solve(n_bits, k_sparse, n_influence_samples=n_influence_samples,
                     n_verify_samples=100, seed=seed, tracker=tracker)
        results.append(r)

        if verbose:
            status = "CORRECT" if r['correct'] else "WRONG"
            print(f"    seed={seed}: {status}  candidates={r['n_candidates']}  "
                  f"subsets_checked={r['n_subsets_checked']}/{c_n_k}  "
                  f"queries={r['total_queries']:,}  time={r['time_total_s']:.4f}s")
            if use_tracker:
                tracker.report()

    return results


def main():
    print("=" * 70)
    print("  EXPERIMENT: Kushilevitz-Mansour Algorithm for Sparse Parity")
    print("  Influence-based pruning — blank slate, no neural net")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # -------------------------------------------------------------------
    # Config 1: n=20, k=3 — C(20,3) = 1,140
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=20, k=3  [C(20,3) = 1,140]")
    print("=" * 70)
    all_results['n20_k3'] = run_config(
        n_bits=20, k_sparse=3, n_influence_samples=200, seeds=seeds
    )

    # -------------------------------------------------------------------
    # Config 2: n=50, k=3 — C(50,3) = 19,600
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=50, k=3  [C(50,3) = 19,600]")
    print("=" * 70)
    all_results['n50_k3'] = run_config(
        n_bits=50, k_sparse=3, n_influence_samples=200, seeds=seeds
    )

    # -------------------------------------------------------------------
    # Config 3: n=100, k=3 — C(100,3) = 161,700
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 3: n=100, k=3  [C(100,3) = 161,700]")
    print("=" * 70)
    all_results['n100_k3'] = run_config(
        n_bits=100, k_sparse=3, n_influence_samples=200, seeds=seeds
    )

    # -------------------------------------------------------------------
    # Config 4: n=20, k=5 — C(20,5) = 15,504
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 4: n=20, k=5  [C(20,5) = 15,504]")
    print("=" * 70)
    all_results['n20_k5'] = run_config(
        n_bits=20, k_sparse=5, n_influence_samples=200, seeds=seeds
    )

    # -------------------------------------------------------------------
    # Sample complexity sweep
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SAMPLE COMPLEXITY: How few influence samples are needed?")
    print("=" * 70)

    all_results['sample_complexity_n20_k3'] = sample_complexity_sweep(20, 3, seed=42)
    all_results['sample_complexity_n50_k3'] = sample_complexity_sweep(50, 3, seed=42)
    all_results['sample_complexity_n20_k5'] = sample_complexity_sweep(20, 5, seed=42)

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    header = (f"  {'Config':<12} | {'C(n,k)':>10} | {'Candidates':>10} | "
              f"{'Subsets':>8} | {'Queries':>10} | {'Time':>8} | {'Correct':>7}")
    print(header)
    print("  " + "-" * 88)

    for key in ['n20_k3', 'n50_k3', 'n100_k3', 'n20_k5']:
        runs = all_results[key]
        n_correct = sum(1 for r in runs if r['correct'])
        avg_cand = np.mean([r['n_candidates'] for r in runs])
        avg_subsets = np.mean([r['n_subsets_checked'] for r in runs])
        avg_queries = np.mean([r['total_queries'] for r in runs])
        avg_time = np.mean([r['time_total_s'] for r in runs])
        c_n_k = runs[0]['c_n_k']

        print(f"  {key:<12} | {c_n_k:>10,} | {avg_cand:>10.1f} | "
              f"{avg_subsets:>8.1f} | {avg_queries:>10,.0f} | {avg_time:>7.4f}s | "
              f"{n_correct}/{len(runs):>5}")

    # -------------------------------------------------------------------
    # Comparison: KM vs Fourier vs Random vs Evolutionary
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  COMPARISON: KM vs Other Approaches")
    print("=" * 90)
    print(f"  {'Config':<12} | {'Method':<18} | {'Queries/Tries':>14} | {'Time':>10} | {'Correct':>7}")
    print("  " + "-" * 85)

    for key in ['n20_k3', 'n50_k3', 'n20_k5']:
        runs = all_results[key]
        avg_queries = np.mean([r['total_queries'] for r in runs])
        avg_time = np.mean([r['time_total_s'] for r in runs])
        n_correct = sum(1 for r in runs if r['correct'])
        c_n_k = runs[0]['c_n_k']

        print(f"  {key:<12} | {'KM (this)':<18} | {avg_queries:>14,.0f} | {avg_time:>9.4f}s | "
              f"{n_correct}/{len(runs)}")

    # Baselines from other experiments
    print(f"  {'n20_k3':<12} | {'Fourier':<18} | {'1,140 subsets':>14} | {'0.009s':>10} | {'3/3'}")
    print(f"  {'n50_k3':<12} | {'Fourier':<18} | {'19,600 subsets':>14} | {'0.16s':>10} | {'3/3'}")
    print(f"  {'n20_k5':<12} | {'Fourier':<18} | {'15,504 subsets':>14} | {'0.14s':>10} | {'3/3'}")
    print(f"  {'n20_k3':<12} | {'Random search':<18} | {'~881 tries':>14} | {'0.011s':>10} | {'5/5'}")
    print(f"  {'n50_k3':<12} | {'Random search':<18} | {'~11,291 tries':>14} | {'0.142s':>10} | {'5/5'}")
    print(f"  {'n20_k3':<12} | {'SGD (fast.py)':<18} | {'~1000 samples':>14} | {'0.12s':>10} | {'yes'}")
    print(f"  {'n50_k3':<12} | {'SGD direct':<18} | {'---':>14} | {'---':>10} | {'FAIL'}")
    print("=" * 90)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_km'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_km',
            'description': 'Kushilevitz-Mansour algorithm — influence-based pruning for sparse parity',
            'hypothesis': 'Influence estimation identifies secret bits with O(n) queries, '
                          'pruning search from C(n,k) to C(k,k)=1',
            'configs': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

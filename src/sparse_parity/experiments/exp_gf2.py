#!/usr/bin/env python3
"""
Experiment: Gaussian Elimination over GF(2) for Sparse Parity

Hypothesis: Treating each sample as a linear equation over GF(2) (binary field),
Gaussian elimination recovers the secret parity bits in O(n^2) time with only
n+1 samples. This is the theoretically optimal approach for pure parity.

The key insight: parity over {-1,+1} is equivalent to XOR over {0,1}.
  - Convert inputs: x_bit = (x+1)/2  so -1->0, +1->1
  - Convert labels: y_bit = (y+1)/2  so -1->0, +1->1
  - For odd k:  y_bit = XOR(x_bit[secret]) = sum(x_bit[secret]) mod 2
  - For even k: y_bit = 1 - XOR(x_bit[secret]) (the relationship is inverted)
  - Since we don't know k's parity a priori, we solve BOTH A*s=b and A*s=(1-b)
  - This is a linear system over GF(2): Gaussian elimination finds s
  - The solution vector has 1s at the secret bit positions.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_gf2.py
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
    """Generate sparse parity data. Returns x, y, secret."""
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    return x, y, secret


# =============================================================================
# GF(2) GAUSSIAN ELIMINATION
# =============================================================================

def gf2_gauss_elim(A, b):
    """
    Solve A * s = b over GF(2) using Gaussian elimination with partial pivoting.

    A: (m x n) binary matrix (numpy uint8, values 0 or 1)
    b: (m,) binary vector

    Returns:
        solution: (n,) binary vector s such that A*s = b (mod 2), or None if inconsistent
        rank: rank of the augmented matrix
    """
    m, n = A.shape
    # Build augmented matrix [A | b]
    aug = np.zeros((m, n + 1), dtype=np.uint8)
    aug[:, :n] = A
    aug[:, n] = b

    pivot_row = 0
    pivot_cols = []

    for col in range(n):
        # Find a row with a 1 in this column at or below pivot_row
        found = -1
        for row in range(pivot_row, m):
            if aug[row, col] == 1:
                found = row
                break

        if found == -1:
            continue  # no pivot in this column, skip

        # Swap rows
        if found != pivot_row:
            aug[[pivot_row, found]] = aug[[found, pivot_row]]

        pivot_cols.append(col)

        # Eliminate all other rows that have a 1 in this column
        for row in range(m):
            if row != pivot_row and aug[row, col] == 1:
                aug[row] = aug[row] ^ aug[pivot_row]  # XOR = addition mod 2

        pivot_row += 1

    rank = pivot_row

    # Check consistency: any row with all-zero A part but non-zero b part
    for row in range(rank, m):
        if aug[row, n] == 1:
            return None, rank  # inconsistent

    # Back-substitute to get solution
    # Free variables get 0 (we want the sparsest solution)
    solution = np.zeros(n, dtype=np.uint8)
    for i, col in enumerate(pivot_cols):
        solution[col] = aug[i, n]

    return solution, rank


def gf2_solve(x, y, n_bits, tracker=None):
    """
    Convert {-1,+1} data to GF(2) and solve with Gaussian elimination.

    For odd k: y_bit = XOR(x_bit[S]), so solve A*s = b
    For even k: y_bit = 1 - XOR(x_bit[S]), so solve A*s = (1-b)
    Since k is unknown, we try both and return whichever yields a valid solution.

    Returns (predicted_secret, solution_vector, rank).
    """
    n_samples = x.shape[0]

    # Convert to GF(2)
    # x_bit: -1 -> 0, +1 -> 1
    A = ((x + 1) / 2).astype(np.uint8)
    # y_bit: -1 -> 0, +1 -> 1
    b = ((y + 1) / 2).astype(np.uint8)

    if tracker:
        tracker.write('x_input', x.size)
        tracker.write('A_gf2', A.size)
        tracker.write('b_gf2', n_samples)
        tracker.read('x_input', x.size)  # read to convert

    if tracker:
        tracker.read('A_gf2', A.size)
        tracker.read('b_gf2', n_samples)

    # Try both b (odd k) and 1-b (even k)
    solutions = []
    for b_try in [b, (1 - b).astype(np.uint8)]:
        solution, rank = gf2_gauss_elim(A.copy(), b_try.copy())
        if solution is not None:
            predicted = sorted(np.where(solution == 1)[0].tolist())
            solutions.append((predicted, solution, rank))

    if tracker:
        tracker.write('solution', n_bits)

    if not solutions:
        return None, None, rank

    # If both solutions exist, verify which one actually produces correct labels
    # on the training data (using the original {-1,+1} representation)
    for predicted, solution, rank in solutions:
        if len(predicted) > 0:
            y_check = np.prod(x[:, predicted], axis=1)
            if np.all(y_check == y):
                return predicted, solution, rank

    # Fallback: return the first solution found
    predicted, solution, rank = solutions[0]
    return predicted, solution, rank


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_config(n_bits, k_sparse, n_samples_list, seeds, verbose=True):
    """Run GF(2) solver on one (n, k) config with varying sample counts."""
    c_n_k = comb(n_bits, k_sparse)
    if verbose:
        print(f"\n  Config: n={n_bits}, k={k_sparse}, C(n,k)={c_n_k:,}")

    results = []

    for n_samples in n_samples_list:
        seed_results = []
        for seed in seeds:
            x, y, secret = generate_data(n_bits, k_sparse, n_samples, seed=seed)

            use_tracker = (seed == seeds[0] and n_samples == n_samples_list[0])
            tracker = MemTracker() if use_tracker else None

            start = time.time()
            predicted, solution, rank = gf2_solve(x, y, n_bits, tracker=tracker)
            elapsed = time.time() - start

            correct = (predicted == secret) if predicted is not None else False

            # Verify on test data
            if predicted is not None and len(predicted) > 0:
                rng_te = np.random.RandomState(seed + 1000)
                x_te = rng_te.choice([-1.0, 1.0], size=(1000, n_bits))
                y_te = np.prod(x_te[:, secret], axis=1)
                y_pred = np.prod(x_te[:, predicted], axis=1)
                test_acc = float(np.mean(y_pred == y_te))
            else:
                test_acc = 0.0

            seed_result = {
                'seed': seed,
                'n_samples': n_samples,
                'secret': secret,
                'predicted': predicted,
                'correct': correct,
                'test_acc': round(test_acc, 4),
                'rank': int(rank),
                'elapsed_s': round(elapsed, 8),
                'k_found': len(predicted) if predicted is not None else 0,
            }

            if tracker:
                seed_result['tracker'] = tracker.to_json()

            seed_results.append(seed_result)

        n_correct = sum(1 for r in seed_results if r['correct'])
        avg_time = np.mean([r['elapsed_s'] for r in seed_results])
        avg_test_acc = np.mean([r['test_acc'] for r in seed_results])

        if verbose:
            status = f"{n_correct}/{len(seeds)} correct" if n_correct > 0 else "ALL FAILED"
            print(f"    n_samples={n_samples:>5}: {status}, "
                  f"avg time={avg_time*1e6:.1f}us, avg test_acc={avg_test_acc:.0%}")

        results.append({
            'n_samples': n_samples,
            'n_correct': n_correct,
            'n_total': len(seeds),
            'avg_time_s': round(float(avg_time), 8),
            'avg_time_us': round(float(avg_time * 1e6), 2),
            'avg_test_acc': round(float(avg_test_acc), 4),
            'per_seed': seed_results,
        })

        # Print tracker report for first config
        if use_tracker and tracker:
            tracker.report()

    return {
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'c_n_k': c_n_k,
        'results': results,
    }


def main():
    print("=" * 70)
    print("  EXPERIMENT: Gaussian Elimination over GF(2)")
    print("  Theoretically optimal for pure parity: O(n^2), microseconds")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # Sample counts to test
    sample_counts_small = [21, 40, 50, 100, 500]   # n+1 = 21 for n=20
    sample_counts_medium = [51, 100, 200, 500]      # n+1 = 51 for n=50
    sample_counts_large = [101, 200, 500]            # n+1 = 101 for n=100

    # -------------------------------------------------------------------
    # Config 1: n=20, k=3
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=20, k=3")
    print("=" * 70)
    all_results['n20_k3'] = run_config(
        n_bits=20, k_sparse=3,
        n_samples_list=sample_counts_small,
        seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 2: n=50, k=3
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=50, k=3")
    print("=" * 70)
    all_results['n50_k3'] = run_config(
        n_bits=50, k_sparse=3,
        n_samples_list=sample_counts_medium,
        seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 3: n=100, k=3
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 3: n=100, k=3")
    print("=" * 70)
    all_results['n100_k3'] = run_config(
        n_bits=100, k_sparse=3,
        n_samples_list=sample_counts_large,
        seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 4: n=20, k=5
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 4: n=20, k=5")
    print("=" * 70)
    all_results['n20_k5'] = run_config(
        n_bits=20, k_sparse=5,
        n_samples_list=sample_counts_small,
        seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 5: n=20, k=7
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 5: n=20, k=7")
    print("=" * 70)
    all_results['n20_k7'] = run_config(
        n_bits=20, k_sparse=7,
        n_samples_list=sample_counts_small,
        seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Config 6: n=20, k=10
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 6: n=20, k=10")
    print("=" * 70)
    all_results['n20_k10'] = run_config(
        n_bits=20, k_sparse=10,
        n_samples_list=sample_counts_small,
        seeds=seeds,
    )

    # -------------------------------------------------------------------
    # Sample complexity deep dive: n=20, k=3
    # How few samples can we get away with?
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SAMPLE COMPLEXITY: n=20, k=3 -- minimum samples needed?")
    print("=" * 70)
    sample_complexity_results = []
    for n_samp in [5, 10, 15, 18, 19, 20, 21, 22, 25, 30, 40, 50, 100]:
        correct_count = 0
        total_time = 0
        for seed in seeds:
            x, y, secret = generate_data(20, 3, n_samp, seed=seed)
            start = time.time()
            predicted, _, rank = gf2_solve(x, y, 20)
            elapsed = time.time() - start
            if predicted == secret:
                correct_count += 1
            total_time += elapsed
        avg_time = total_time / len(seeds)
        print(f"    n_samples={n_samp:>4}: {correct_count}/{len(seeds)} correct, "
              f"avg {avg_time*1e6:.1f}us")
        sample_complexity_results.append({
            'n_samples': n_samp,
            'correct': correct_count,
            'total': len(seeds),
            'avg_time_us': round(avg_time * 1e6, 2),
        })
    all_results['sample_complexity'] = sample_complexity_results

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    header = (f"  {'Config':<15} | {'C(n,k)':>10} | {'n_samp':>7} | "
              f"{'Correct':>7} | {'Avg Time':>12} | {'Test Acc':>8}")
    print(header)
    print("  " + "-" * 75)

    for key, res in all_results.items():
        if key == 'sample_complexity':
            continue
        n_b = res['n_bits']
        k_s = res['k_sparse']
        c = res['c_n_k']
        # Show best result (highest n_samples)
        best = res['results'][-1]  # last entry = most samples
        print(f"  n={n_b},k={k_s:<8} | {c:>10,} | {best['n_samples']:>7} | "
              f"{best['n_correct']}/{best['n_total']:>5} | "
              f"{best['avg_time_us']:>9.1f} us | {best['avg_test_acc']:>7.0%}")

    # -------------------------------------------------------------------
    # Comparison with other approaches
    # -------------------------------------------------------------------
    print("\n  " + "=" * 85)
    print("  COMPARISON WITH OTHER APPROACHES")
    print("  " + "=" * 85)
    print(f"  {'Config':<20} | {'Method':<20} | {'Time':>12} | {'Samples':>8} | {'Notes'}")
    print("  " + "-" * 80)

    # GF(2) results
    for key in ['n20_k3', 'n50_k3', 'n100_k3', 'n20_k5', 'n20_k7', 'n20_k10']:
        if key not in all_results:
            continue
        res = all_results[key]
        best = res['results'][-1]
        # Also show minimum working sample count
        min_working = None
        for r in res['results']:
            if r['n_correct'] == r['n_total']:
                min_working = r['n_samples']
                break
        min_str = f"min={min_working}" if min_working else "partial"
        print(f"  {key:<20} | {'GF(2) Gauss':<20} | "
              f"{best['avg_time_us']:>9.1f} us | {best['n_samples']:>8} | "
              f"{min_str}")

    print(f"  {'n20_k3':<20} | {'SGD baseline':<20} | {'120,000 us':>12} | {'10000':>8} | ~5 epochs")
    print(f"  {'n50_k3':<20} | {'SGD (curriculum)':<20} | {'---':>12} | {'10000':>8} | 20 epochs")
    print(f"  {'n50_k3':<20} | {'SGD (direct)':<20} | {'---':>12} | {'10000':>8} | FAIL (54%)")
    print(f"  {'n20_k3':<20} | {'Fourier exh.':<20} | {'~3,000 us':>12} | {'500':>8} | C(20,3)=1140 subsets")
    print(f"  {'n20_k3':<20} | {'Random search':<20} | {'~11,000 us':>12} | {'500':>8} | ~881 tries")
    print("=" * 90)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_gf2'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_gf2',
            'description': 'Gaussian Elimination over GF(2) for sparse parity',
            'hypothesis': 'GF(2) Gaussian elimination solves sparse parity in O(n^2) with n+1 samples',
            'approach': 'blank_slate -- no neural net, no SGD, no gradients',
            'configs': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Experiment: SMT/Constraint Solver for Sparse Parity

Hypothesis: Sparse parity can be encoded as an SMT constraint satisfaction
problem. For each training sample, the constraint is:
    prod(x[idx_0], x[idx_1], ..., x[idx_{k-1}]) == y
With ordered indices idx_0 < idx_1 < ... < idx_{k-1}, this has a unique
solution (the secret subset). The search space is C(n,k), which is tiny
for modern SMT solvers.

This approach uses Z3 if available, otherwise falls back to a backtracking
constraint solver.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_smt.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from math import comb
from itertools import combinations

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
# APPROACH 1: Z3 SMT SOLVER (Boolean encoding)
# =============================================================================

def smt_solve(x, y, n_bits, k_sparse, max_z3_samples=50):
    """
    Encode sparse parity as an SMT problem using Z3.

    Boolean encoding: sel_j = True means column j is selected.
    Constraints:
      - Exactly k of the sel_j are True (cardinality constraint)
      - For each sample i: the product of x[i,j] for selected j equals y[i]
        Encoded as: count of j where sel_j=True AND x[i,j]==-1 has
        even parity iff y[i]==+1.

    We use a subset of training samples (max_z3_samples) for constraint
    building to keep solver time reasonable.

    Returns (found_indices, solve_time, n_constraints, status).
    """
    from z3 import Bool, Solver, And, Or, Not, Xor, If, Sum, sat, PbEq

    solver = Solver()
    n_samples = min(x.shape[0], max_z3_samples)

    # Boolean selection variables: sel[j] = True if column j is in the subset
    sel = [Bool(f'sel_{j}') for j in range(n_bits)]

    # Cardinality constraint: exactly k variables are True
    # PbEq encodes pseudo-boolean equality: sum(sel_j) == k
    solver.add(PbEq([(s, 1) for s in sel], k_sparse))
    n_constraints = 1

    # Sample constraints using parity encoding
    # For sample i: product of x[i, selected_j] == y[i]
    # Since x values are {-1, +1}, product = (-1)^(count of -1 values among selected)
    # So: y[i] == +1 iff count of (sel_j AND x[i,j]==-1) is even
    #     y[i] == -1 iff count of (sel_j AND x[i,j]==-1) is odd
    #
    # We encode this with XOR chain over "is negative" indicators.
    # neg_j_i = sel[j] AND (x[i,j] == -1)
    # XOR of all neg_j_i must equal (y[i] == -1)

    for i in range(n_samples):
        # For each column j, it contributes a -1 factor iff sel[j] AND x[i,j]==-1
        neg_indicators = []
        for j in range(n_bits):
            if x[i, j] == -1.0:
                neg_indicators.append(sel[j])
            # If x[i,j] == +1, selecting column j doesn't contribute a -1

        # The parity of neg_indicators must equal (y[i] == -1)
        target_odd = (y[i] == -1.0)

        if len(neg_indicators) == 0:
            # No columns have -1 for this sample, so product is always +1
            if target_odd:
                solver.add(False)  # Impossible
            # else: always satisfied, no constraint needed
        else:
            # Build XOR chain
            parity_expr = neg_indicators[0]
            for ni in neg_indicators[1:]:
                parity_expr = Xor(parity_expr, ni)

            if target_odd:
                solver.add(parity_expr)
            else:
                solver.add(Not(parity_expr))

        n_constraints += 1

    start = time.time()
    result = solver.check()
    solve_time = time.time() - start

    if result == sat:
        model = solver.model()
        found = sorted([j for j in range(n_bits) if model[sel[j]]])
        return found, solve_time, n_constraints, 'SAT'
    else:
        return None, solve_time, n_constraints, 'UNSAT'


# =============================================================================
# APPROACH 2: BACKTRACKING CONSTRAINT SOLVER
# =============================================================================

def backtrack_solve(x, y, n_bits, k_sparse):
    """
    Backtracking search over ordered k-subsets of {0, ..., n_bits-1}.

    For each partial assignment of indices, check consistency with all samples.
    Prune branches that violate any sample.

    Returns (found_indices, solve_time, nodes_explored, status).
    """
    n_samples = x.shape[0]
    start = time.time()
    nodes_explored = 0

    def backtrack(chosen, start_from):
        nonlocal nodes_explored
        nodes_explored += 1

        m = len(chosen)

        if m == k_sparse:
            # Full assignment: check exact match
            parity = np.prod(x[:, chosen], axis=1)
            if np.all(parity == y):
                return list(chosen)
            return None

        if m == k_sparse - 1:
            # One index left: compute target column and look for it
            partial_product = np.prod(x[:, chosen], axis=1)
            target = y * partial_product  # what the last column must equal
            for col in range(chosen[-1] + 1 if chosen else start_from, n_bits):
                if np.all(x[:, col] == target):
                    return list(chosen) + [col]
            return None

        # General case: try each next index
        for idx in range(start_from, n_bits - (k_sparse - m - 1)):
            chosen.append(idx)
            result = backtrack(chosen, idx + 1)
            if result is not None:
                return result
            chosen.pop()

        return None

    found = backtrack([], 0)
    solve_time = time.time() - start

    if found is not None:
        return sorted(found), solve_time, nodes_explored, 'FOUND'
    else:
        return None, solve_time, nodes_explored, 'NOT_FOUND'


# =============================================================================
# SAMPLE COMPLEXITY TEST
# =============================================================================

def test_sample_complexity(n_bits, k_sparse, seed=42, max_samples=200):
    """
    Find minimum number of training samples for unique solution.
    Start with small sample counts and increase until the solver finds the
    correct (unique) solution consistently.

    Returns list of dicts with n_samples, solved, correct, solve_time.
    """
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())

    # Generate a large pool of samples
    x_pool = rng.choice([-1.0, 1.0], size=(max_samples, n_bits))
    y_pool = np.prod(x_pool[:, secret], axis=1)

    results = []
    sample_counts = list(range(k_sparse, min(4 * k_sparse + 2, max_samples), 1))
    sample_counts.extend([5 * k_sparse, 10 * k_sparse, 20 * k_sparse, 50 * k_sparse])
    sample_counts = sorted(set(s for s in sample_counts if s <= max_samples))

    for n_s in sample_counts:
        x_sub = x_pool[:n_s]
        y_sub = y_pool[:n_s]

        found, solve_time, nodes, status = backtrack_solve(x_sub, y_sub, n_bits, k_sparse)
        solved = found is not None
        correct = found == secret if solved else False

        results.append({
            'n_samples': n_s,
            'solved': solved,
            'correct': correct,
            'found': found,
            'secret': secret,
            'solve_time': round(solve_time, 6),
            'nodes_explored': nodes,
        })

    return results


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_config(n_bits, k_sparse, n_train, seeds, use_z3=True, verbose=True):
    """Run SMT and backtracking on one config across multiple seeds."""
    c_n_k = comb(n_bits, k_sparse)
    if verbose:
        print(f"\n  Config: n={n_bits}, k={k_sparse}, C(n,k)={c_n_k:,}")
        print(f"  Training samples: {n_train}")

    z3_results = []
    bt_results = []

    for seed in seeds:
        x, y, secret = generate_data(n_bits, k_sparse, n_train, seed=seed)
        n_test = 500
        x_test, y_test, _ = generate_data(n_bits, k_sparse, n_test, seed=seed + 1000)
        # Recompute y_test with the secret from this seed's data
        y_test = np.prod(x_test[:, secret], axis=1)

        # Memory tracking
        tracker = MemTracker()
        tracker.write('x_train', n_train * n_bits)
        tracker.write('y_train', n_train)
        tracker.read('x_train', n_train * n_bits)
        tracker.read('y_train', n_train)

        # --- Z3 SMT Solver ---
        if use_z3:
            try:
                # Use a limited number of samples for Z3 encoding
                z3_samples = min(n_train, 50)
                build_start = time.time()
                found_z3, z3_time, n_constraints, z3_status = smt_solve(
                    x[:z3_samples], y[:z3_samples], n_bits, k_sparse,
                    max_z3_samples=z3_samples
                )
                total_z3_time = time.time() - build_start
                solved_z3 = found_z3 is not None

                # Verify on full training set
                if solved_z3:
                    y_pred_train = np.prod(x[:, found_z3], axis=1)
                    train_acc = float(np.mean(y_pred_train == y))
                    if train_acc < 1.0:
                        # Solution from subset doesn't work on full set
                        solved_z3 = False
                        z3_status = 'SAT_BUT_WRONG'

                correct_z3 = found_z3 == secret if solved_z3 else False

                # Test accuracy
                if solved_z3:
                    y_pred = np.prod(x_test[:, found_z3], axis=1)
                    test_acc = float(np.mean(y_pred == y_test))
                else:
                    test_acc = 0.0

                z3_results.append({
                    'seed': seed,
                    'found': found_z3,
                    'secret': secret,
                    'correct': correct_z3,
                    'solved': solved_z3,
                    'solve_time_s': round(z3_time, 6),
                    'total_time_s': round(total_z3_time, 6),
                    'n_constraints': n_constraints,
                    'z3_samples_used': z3_samples,
                    'status': z3_status,
                    'test_accuracy': test_acc,
                })
                if verbose:
                    status_str = (
                        f"SOLVED in {total_z3_time:.4f}s "
                        f"(solve={z3_time:.4f}s, {n_constraints} constraints, "
                        f"{z3_samples} samples)"
                        if solved_z3 else f"FAILED ({z3_status})")
                    print(f"    [Z3]         seed={seed}: {status_str}")
            except Exception as e:
                z3_results.append({
                    'seed': seed,
                    'error': str(e),
                    'solved': False,
                })
                if verbose:
                    print(f"    [Z3]         seed={seed}: ERROR - {e}")

        # --- Backtracking Solver ---
        found_bt, bt_time, nodes, bt_status = backtrack_solve(
            x, y, n_bits, k_sparse
        )
        solved_bt = found_bt is not None
        correct_bt = found_bt == secret if solved_bt else False

        if solved_bt:
            y_pred = np.prod(x_test[:, found_bt], axis=1)
            test_acc_bt = float(np.mean(y_pred == y_test))
        else:
            test_acc_bt = 0.0

        tracker.write('solution', k_sparse)

        bt_results.append({
            'seed': seed,
            'found': found_bt,
            'secret': secret,
            'correct': correct_bt,
            'solved': solved_bt,
            'solve_time_s': round(bt_time, 6),
            'nodes_explored': nodes,
            'status': bt_status,
            'test_accuracy': test_acc_bt,
            'memory_tracker': tracker.to_json(),
        })
        if verbose:
            status_str = (f"SOLVED in {bt_time:.6f}s ({nodes} nodes)"
                          if solved_bt else f"FAILED ({bt_status})")
            print(f"    [Backtrack]  seed={seed}: {status_str}")

    return {
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'c_n_k': c_n_k,
        'n_train': n_train,
        'z3': z3_results,
        'backtrack': bt_results,
    }


def main():
    print("=" * 70)
    print("  EXPERIMENT: SMT/Constraint Solver for Sparse Parity")
    print("  Approach #15: Program Synthesis / SMT Solver")
    print("=" * 70)

    # Check Z3 availability
    use_z3 = False
    try:
        import z3
        print(f"\n  Z3 solver available (version {z3.get_version_string()})")
        use_z3 = True
    except ImportError:
        print("\n  Z3 not available, using backtracking solver only")

    seeds = [42, 43, 44]
    all_results = {}

    # -------------------------------------------------------------------
    # Config 1: n=20, k=3 -- C(20,3) = 1140
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=20, k=3  [C(20,3) = 1,140]")
    print("=" * 70)
    all_results['n20_k3'] = run_config(
        n_bits=20, k_sparse=3, n_train=500, seeds=seeds, use_z3=use_z3
    )

    # -------------------------------------------------------------------
    # Config 2: n=50, k=3 -- C(50,3) = 19600
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=50, k=3  [C(50,3) = 19,600]")
    print("=" * 70)
    all_results['n50_k3'] = run_config(
        n_bits=50, k_sparse=3, n_train=500, seeds=seeds, use_z3=use_z3
    )

    # -------------------------------------------------------------------
    # Config 3: n=100, k=3 -- C(100,3) = 161700
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 3: n=100, k=3  [C(100,3) = 161,700]")
    print("=" * 70)
    all_results['n100_k3'] = run_config(
        n_bits=100, k_sparse=3, n_train=500, seeds=seeds, use_z3=use_z3
    )

    # -------------------------------------------------------------------
    # Config 4: n=20, k=5 -- C(20,5) = 15504
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 4: n=20, k=5  [C(20,5) = 15,504]")
    print("=" * 70)
    all_results['n20_k5'] = run_config(
        n_bits=20, k_sparse=5, n_train=2000, seeds=seeds, use_z3=use_z3
    )

    # -------------------------------------------------------------------
    # Sample complexity test
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SAMPLE COMPLEXITY TEST: n=20, k=3")
    print("  How many samples needed for unique solution?")
    print("=" * 70)

    sample_complexity_results = test_sample_complexity(
        n_bits=20, k_sparse=3, seed=42, max_samples=200
    )
    min_unique = None
    for r in sample_complexity_results:
        status = "CORRECT" if r['correct'] else ("WRONG" if r['solved'] else "NO SOLUTION")
        print(f"    samples={r['n_samples']:>3}: {status} "
              f"({r['solve_time']:.6f}s, {r['nodes_explored']} nodes)")
        if r['correct'] and min_unique is None:
            min_unique = r['n_samples']

    if min_unique is not None:
        print(f"\n  Minimum samples for unique correct solution: {min_unique}")

    all_results['sample_complexity'] = {
        'n_bits': 20,
        'k_sparse': 3,
        'min_unique_samples': min_unique,
        'details': sample_complexity_results,
    }

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)

    configs = ['n20_k3', 'n50_k3', 'n100_k3', 'n20_k5']
    header = (f"  {'Config':<12} | {'C(n,k)':>8} | "
              f"{'Z3 time':>10} | {'Z3 solved':>9} | "
              f"{'BT time':>10} | {'BT nodes':>9} | {'BT solved':>9} | "
              f"{'Test acc':>8}")
    print(header)
    print("  " + "-" * 88)

    for key in configs:
        res = all_results[key]

        # Z3 stats
        if res['z3']:
            z3_solved = sum(1 for r in res['z3'] if r.get('solved', False))
            z3_times = [r['total_time_s'] for r in res['z3'] if r.get('solved', False)]
            z3_avg = np.mean(z3_times) if z3_times else float('nan')
            z3_str = f"{z3_avg:.4f}s" if z3_times else "---"
            z3_s_str = f"{z3_solved}/{len(res['z3'])}"
        else:
            z3_str = "N/A"
            z3_s_str = "N/A"

        # Backtracking stats
        bt_solved = sum(1 for r in res['backtrack'] if r.get('solved', False))
        bt_times = [r['solve_time_s'] for r in res['backtrack'] if r.get('solved', False)]
        bt_nodes_list = [r['nodes_explored'] for r in res['backtrack'] if r.get('solved', False)]
        bt_avg = np.mean(bt_times) if bt_times else float('nan')
        bt_nodes_avg = np.mean(bt_nodes_list) if bt_nodes_list else float('nan')
        bt_str = f"{bt_avg:.6f}s" if bt_times else "---"
        bt_n_str = f"{bt_nodes_avg:.0f}" if bt_nodes_list else "---"
        bt_s_str = f"{bt_solved}/{len(res['backtrack'])}"

        # Test accuracy
        bt_accs = [r['test_accuracy'] for r in res['backtrack'] if r.get('solved', False)]
        acc_str = f"{np.mean(bt_accs):.1%}" if bt_accs else "---"

        print(f"  {key:<12} | {res['c_n_k']:>8,} | "
              f"{z3_str:>10} | {z3_s_str:>9} | "
              f"{bt_str:>10} | {bt_n_str:>9} | {bt_s_str:>9} | "
              f"{acc_str:>8}")

    # -------------------------------------------------------------------
    # Memory tracker report (last config)
    # -------------------------------------------------------------------
    print("\n  Memory Tracker Report (last backtrack run, n=20/k=5):")
    last_bt = all_results['n20_k5']['backtrack'][-1]
    if 'memory_tracker' in last_bt:
        mt = last_bt['memory_tracker']
        print(f"    Total floats accessed: {mt.get('total_floats_accessed', 'N/A'):,}")
        print(f"    Reads: {mt.get('reads', 'N/A')}, Writes: {mt.get('writes', 'N/A')}")
        print(f"    Weighted ARD: {mt.get('weighted_ard', 'N/A'):,.0f}")

    print("=" * 90)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_smt'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_smt',
            'description': 'SMT/Constraint solver for sparse parity (Z3 + backtracking)',
            'hypothesis': 'Sparse parity encoded as constraints is trivial for SMT solvers',
            'approach': 'Program Synthesis / SMT Solver (#15)',
            'z3_available': use_z3,
            'configs': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

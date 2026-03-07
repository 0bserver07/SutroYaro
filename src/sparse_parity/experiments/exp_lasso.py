#!/usr/bin/env python3
"""
Experiment: LASSO on Interaction Features for Sparse Parity

Hypothesis: If we expand input x in {-1,+1}^n to all C(n,k) interaction
terms (x_i * x_j * x_k for k=3), then run L1-penalized linear regression
(LASSO), the true solution has exactly 1 nonzero coefficient.
LASSO's sparsity assumption is a perfect match because the parity function
IS linear in the interaction basis.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_lasso.py
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

from sklearn.linear_model import Lasso, LassoCV


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


# =============================================================================
# FEATURE EXPANSION
# =============================================================================

def expand_interactions(x, n_bits, k_sparse, tracker=None):
    """
    Expand x (n_samples, n_bits) to all C(n,k) interaction features.
    Each feature is the product of k input columns.

    Returns (X_expanded, subset_list) where subset_list[j] is the k-tuple
    corresponding to column j of X_expanded.
    """
    n_samples = x.shape[0]
    n_features = comb(n_bits, k_sparse)
    subset_list = list(combinations(range(n_bits), k_sparse))

    if tracker:
        tracker.write('x_raw', x.size)

    X_expanded = np.empty((n_samples, n_features), dtype=np.float64)

    for j, subset in enumerate(subset_list):
        if tracker:
            tracker.read('x_raw', n_samples * k_sparse)
        X_expanded[:, j] = np.prod(x[:, list(subset)], axis=1)

    if tracker:
        tracker.write('X_expanded', X_expanded.size)

    return X_expanded, subset_list


# =============================================================================
# LASSO SOLVER
# =============================================================================

def lasso_solve(X_expanded, y, subset_list, alpha=0.1, tracker=None):
    """
    Run LASSO on expanded features. The true parity function has exactly
    one nonzero coefficient (= 1.0) at the column corresponding to the
    secret subset.

    Returns (predicted_subset, coef_info, model).
    """
    if tracker:
        tracker.read('X_expanded', X_expanded.size)
        tracker.read('y', y.size)

    model = Lasso(alpha=alpha, max_iter=10000, tol=1e-6, fit_intercept=False)
    model.fit(X_expanded, y)

    if tracker:
        tracker.write('coefs', model.coef_.size)

    coefs = model.coef_
    nonzero_mask = np.abs(coefs) > 1e-6
    n_nonzero = int(np.sum(nonzero_mask))

    # The predicted subset is the one with the largest |coefficient|
    best_idx = int(np.argmax(np.abs(coefs)))
    predicted_subset = list(subset_list[best_idx])
    best_coef = float(coefs[best_idx])

    coef_info = {
        'n_nonzero': n_nonzero,
        'best_idx': best_idx,
        'best_coef': round(best_coef, 6),
        'max_abs_coef': round(float(np.max(np.abs(coefs))), 6),
        'nonzero_subsets': [list(subset_list[i]) for i in range(len(coefs)) if nonzero_mask[i]],
        'nonzero_coefs': [round(float(coefs[i]), 6) for i in range(len(coefs)) if nonzero_mask[i]],
    }

    return predicted_subset, coef_info, model


def lasso_cv_solve(X_expanded, y, subset_list, tracker=None):
    """
    Run LassoCV (cross-validated alpha selection) on expanded features.
    Returns (predicted_subset, coef_info, model, best_alpha).
    """
    if tracker:
        tracker.read('X_expanded', X_expanded.size)
        tracker.read('y', y.size)

    model = LassoCV(cv=5, max_iter=10000, tol=1e-6, fit_intercept=False, n_jobs=1)
    model.fit(X_expanded, y)

    if tracker:
        tracker.write('coefs_cv', model.coef_.size)

    coefs = model.coef_
    nonzero_mask = np.abs(coefs) > 1e-6
    n_nonzero = int(np.sum(nonzero_mask))

    best_idx = int(np.argmax(np.abs(coefs)))
    predicted_subset = list(subset_list[best_idx])
    best_coef = float(coefs[best_idx])

    coef_info = {
        'n_nonzero': n_nonzero,
        'best_idx': best_idx,
        'best_coef': round(best_coef, 6),
        'max_abs_coef': round(float(np.max(np.abs(coefs))), 6),
        'best_alpha': round(float(model.alpha_), 8),
        'nonzero_subsets': [list(subset_list[i]) for i in range(len(coefs)) if nonzero_mask[i]],
        'nonzero_coefs': [round(float(coefs[i]), 6) for i in range(len(coefs)) if nonzero_mask[i]],
    }

    return predicted_subset, coef_info, model, float(model.alpha_)


# =============================================================================
# SINGLE CONFIG RUNNER
# =============================================================================

def run_config(n_bits, k_sparse, n_samples, seed=42, alpha=0.1,
               use_cv=True, use_tracker=False, verbose=True):
    """Run LASSO on one config. Returns result dict."""
    n_features = comb(n_bits, k_sparse)

    if verbose:
        print(f"\n  n={n_bits}, k={k_sparse}: {n_features:,} interaction features, "
              f"{n_samples} samples, seed={seed}")

    # Generate data
    x, y, secret = generate_data(n_bits, k_sparse, n_samples, seed=seed)

    tracker = MemTracker() if use_tracker else None

    # Expand features
    t0 = time.time()
    X_expanded, subset_list = expand_interactions(x, n_bits, k_sparse, tracker=tracker)
    t_expand = time.time() - t0

    if tracker:
        tracker.write('y', y.size)

    # Run LASSO with fixed alpha
    t1 = time.time()
    predicted, coef_info, model = lasso_solve(
        X_expanded, y, subset_list, alpha=alpha, tracker=tracker
    )
    t_lasso = time.time() - t1

    correct = (predicted == secret)

    # Run LassoCV if requested
    cv_result = None
    t_cv = 0
    if use_cv:
        t2 = time.time()
        pred_cv, coef_cv, model_cv, best_alpha = lasso_cv_solve(
            X_expanded, y, subset_list, tracker=tracker
        )
        t_cv = time.time() - t2
        correct_cv = (pred_cv == secret)
        cv_result = {
            'predicted': pred_cv,
            'correct': correct_cv,
            'best_alpha': round(best_alpha, 8),
            'n_nonzero': coef_cv['n_nonzero'],
            'best_coef': coef_cv['best_coef'],
            'elapsed_s': round(t_cv, 6),
        }

    total_time = t_expand + t_lasso + t_cv

    # Test accuracy on fresh data
    rng_te = np.random.RandomState(seed + 1000)
    x_te = rng_te.choice([-1.0, 1.0], size=(500, n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    y_pred = np.prod(x_te[:, predicted], axis=1)
    te_acc = float(np.mean(y_pred == y_te))

    if verbose:
        status = "CORRECT" if correct else "WRONG"
        print(f"    Secret:    {secret}")
        print(f"    Predicted: {predicted} ({status})")
        print(f"    LASSO coef: {coef_info['best_coef']:.4f}, "
              f"nonzero: {coef_info['n_nonzero']}")
        print(f"    Test acc: {te_acc:.0%}")
        print(f"    Time: expand={t_expand:.4f}s, lasso={t_lasso:.4f}s, "
              f"total={total_time:.4f}s")
        if cv_result:
            cv_status = "CORRECT" if cv_result['correct'] else "WRONG"
            print(f"    LassoCV: alpha={cv_result['best_alpha']:.6f}, "
                  f"nonzero={cv_result['n_nonzero']}, "
                  f"coef={cv_result['best_coef']:.4f} ({cv_status}), "
                  f"time={t_cv:.4f}s")

    tracker_data = None
    if tracker:
        tracker_data = tracker.to_json()
        if verbose:
            tracker.report()

    return {
        'method': 'lasso',
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'n_samples': n_samples,
        'n_features': n_features,
        'alpha': alpha,
        'seed': seed,
        'secret': secret,
        'predicted': predicted,
        'correct': correct,
        'test_acc': round(te_acc, 4),
        'n_nonzero': coef_info['n_nonzero'],
        'best_coef': coef_info['best_coef'],
        'elapsed_expand_s': round(t_expand, 6),
        'elapsed_lasso_s': round(t_lasso, 6),
        'elapsed_total_s': round(total_time, 6),
        'lasso_cv': cv_result,
        'tracker': tracker_data,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: LASSO on Interaction Features for Sparse Parity")
    print("  Expand to C(n,k) interaction terms, then L1 regression")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # -------------------------------------------------------------------
    # CONFIG 1: n=20, k=3 — C(20,3) = 1,140 features
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=20, k=3 (C(20,3) = 1,140 features)")
    print("=" * 70)

    results_20_3 = []
    for seed in seeds:
        r = run_config(20, 3, n_samples=500, seed=seed, alpha=0.1,
                       use_cv=True,
                       use_tracker=(seed == seeds[0]),
                       verbose=(seed == seeds[0]))
        results_20_3.append(r)
        if seed != seeds[0]:
            status = "OK" if r['correct'] else "FAIL"
            cv_status = ""
            if r['lasso_cv']:
                cv_ok = "OK" if r['lasso_cv']['correct'] else "FAIL"
                cv_status = f"  CV: alpha={r['lasso_cv']['best_alpha']:.6f} {cv_ok}"
            print(f"    seed={seed}: {r['elapsed_total_s']:.4f}s  {status}  "
                  f"nonzero={r['n_nonzero']}  coef={r['best_coef']:.4f}{cv_status}")
    all_results['n20_k3'] = results_20_3

    # -------------------------------------------------------------------
    # CONFIG 2: n=50, k=3 — C(50,3) = 19,600 features
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=50, k=3 (C(50,3) = 19,600 features)")
    print("=" * 70)

    results_50_3 = []
    for seed in seeds:
        r = run_config(50, 3, n_samples=1000, seed=seed, alpha=0.1,
                       use_cv=True,
                       use_tracker=False,
                       verbose=(seed == seeds[0]))
        results_50_3.append(r)
        if seed != seeds[0]:
            status = "OK" if r['correct'] else "FAIL"
            cv_status = ""
            if r['lasso_cv']:
                cv_ok = "OK" if r['lasso_cv']['correct'] else "FAIL"
                cv_status = f"  CV: alpha={r['lasso_cv']['best_alpha']:.6f} {cv_ok}"
            print(f"    seed={seed}: {r['elapsed_total_s']:.4f}s  {status}  "
                  f"nonzero={r['n_nonzero']}  coef={r['best_coef']:.4f}{cv_status}")
    all_results['n50_k3'] = results_50_3

    # -------------------------------------------------------------------
    # CONFIG 3: n=20, k=5 — C(20,5) = 15,504 features
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 3: n=20, k=5 (C(20,5) = 15,504 features)")
    print("=" * 70)

    results_20_5 = []
    for seed in seeds:
        r = run_config(20, 5, n_samples=2000, seed=seed, alpha=0.1,
                       use_cv=True,
                       use_tracker=False,
                       verbose=(seed == seeds[0]))
        results_20_5.append(r)
        if seed != seeds[0]:
            status = "OK" if r['correct'] else "FAIL"
            cv_status = ""
            if r['lasso_cv']:
                cv_ok = "OK" if r['lasso_cv']['correct'] else "FAIL"
                cv_status = f"  CV: alpha={r['lasso_cv']['best_alpha']:.6f} {cv_ok}"
            print(f"    seed={seed}: {r['elapsed_total_s']:.4f}s  {status}  "
                  f"nonzero={r['n_nonzero']}  coef={r['best_coef']:.4f}{cv_status}")
    all_results['n20_k5'] = results_20_5

    # -------------------------------------------------------------------
    # ALPHA SWEEP: how sensitive is LASSO to alpha?
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ALPHA SWEEP: n=20, k=3, seed=42")
    print("=" * 70)

    alpha_results = []
    for alpha in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        r = run_config(20, 3, n_samples=500, seed=42, alpha=alpha,
                       use_cv=False, use_tracker=False, verbose=False)
        status = "OK" if r['correct'] else "FAIL"
        print(f"    alpha={alpha:<6}: nonzero={r['n_nonzero']:>5}, "
              f"coef={r['best_coef']:.4f}, time={r['elapsed_total_s']:.4f}s  {status}")
        alpha_results.append({
            'alpha': alpha,
            'correct': r['correct'],
            'n_nonzero': r['n_nonzero'],
            'best_coef': r['best_coef'],
            'elapsed_s': r['elapsed_total_s'],
        })
    all_results['alpha_sweep'] = alpha_results

    # -------------------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    header = (f"  {'Config':<15} | {'Features':>10} | {'Correct':>7} | "
              f"{'Nonzero':>7} | {'Coef':>6} | {'Time':>10} | {'CV alpha':>10}")
    print(header)
    print("  " + "-" * 85)

    for key in ['n20_k3', 'n50_k3', 'n20_k5']:
        runs = all_results[key]
        n_correct = sum(1 for r in runs if r['correct'])
        avg_nonzero = np.mean([r['n_nonzero'] for r in runs])
        avg_coef = np.mean([r['best_coef'] for r in runs])
        avg_time = np.mean([r['elapsed_total_s'] for r in runs])
        n_feat = runs[0]['n_features']

        cv_alphas = [r['lasso_cv']['best_alpha'] for r in runs if r['lasso_cv']]
        avg_cv_alpha = np.mean(cv_alphas) if cv_alphas else float('nan')
        n_cv_correct = sum(1 for r in runs if r['lasso_cv'] and r['lasso_cv']['correct'])

        print(f"  {key:<15} | {n_feat:>10,} | {n_correct}/{len(runs):>5} | "
              f"{avg_nonzero:>7.1f} | {avg_coef:>6.3f} | {avg_time:>9.4f}s | "
              f"{avg_cv_alpha:>10.6f}")

    # -------------------------------------------------------------------
    # COMPARISON WITH BASELINES
    # -------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  COMPARISON: LASSO vs Fourier vs SGD")
    print("=" * 90)
    header = f"  {'Config':<15} | {'Method':<15} | {'Acc':>5} | {'Time':>10} | {'Notes'}"
    print(header)
    print("  " + "-" * 75)

    for key in ['n20_k3', 'n50_k3', 'n20_k5']:
        runs = all_results[key]
        avg_time = np.mean([r['elapsed_total_s'] for r in runs])
        n_correct = sum(1 for r in runs if r['correct'])
        n_feat = runs[0]['n_features']
        print(f"  {key:<15} | {'LASSO':<15} | "
              f"{n_correct}/{len(runs):>3} | {avg_time:>9.4f}s | "
              f"{n_feat:,} features")

    # Baselines from findings
    print(f"  {'n20_k3':<15} | {'Fourier':<15} | {'3/3':>5} | {'0.009s':>10} | 1,140 subsets")
    print(f"  {'n20_k3':<15} | {'SGD':<15} | {'100%':>5} | {'0.12s':>10} | baseline")
    print(f"  {'n20_k3':<15} | {'Random search':<15} | {'5/5':>5} | {'0.011s':>10} | 881 tries avg")
    print(f"  {'n50_k3':<15} | {'Fourier':<15} | {'3/3':>5} | {'0.16s':>10} | 19,600 subsets")
    print(f"  {'n50_k3':<15} | {'SGD direct':<15} | {'54%':>5} | {'---':>10} | FAIL")
    print(f"  {'n20_k5':<15} | {'Fourier':<15} | {'3/3':>5} | {'0.14s':>10} | 15,504 subsets")
    print(f"  {'n20_k5':<15} | {'SGD (n=5000)':<15} | {'100%':>5} | {'---':>10} | 14 epochs")
    print("=" * 90)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_lasso'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_lasso',
            'description': 'LASSO on interaction features for sparse parity',
            'hypothesis': 'L1-penalized regression on C(n,k) interaction features recovers the single true parity term',
            'approach': 'expand to interaction basis, then LASSO',
            'configs': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

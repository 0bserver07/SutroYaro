#!/usr/bin/env python3
"""
Experiment: Decision Trees / Random Forests for Sparse Parity

Hypothesis: A depth-k binary decision tree can learn k-parity exactly if each
split tests one of the secret bits. Random forests with max_depth=k would
implicitly search the subset space. However, greedy splitting by information
gain fails for parity because individual bits have zero marginal correlation
with the label.

We test: DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier
at max_depth=k and max_depth=2*k, across configs n=20/k=3, n=50/k=3, n=20/k=5.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_decision_tree.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sparse_parity.tracker import MemTracker

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n_bits, k_sparse, n_train, n_test=1000, seed=42):
    """Generate sparse parity data. Returns x_train, y_train, x_test, y_test, secret."""
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())

    x_train = rng.choice([-1.0, 1.0], size=(n_train, n_bits))
    y_train = np.prod(x_train[:, secret], axis=1)

    x_test = rng.choice([-1.0, 1.0], size=(n_test, n_bits))
    y_test = np.prod(x_test[:, secret], axis=1)

    return x_train, y_train, x_test, y_test, secret


# =============================================================================
# RUN SINGLE MODEL
# =============================================================================

def run_model(model_name, model, x_train, y_train, x_test, y_test, tracker=None):
    """Fit a model and measure accuracy + time. Returns result dict."""
    # Convert labels from {-1, +1} to {0, 1} for sklearn classifiers
    y_train_01 = ((y_train + 1) / 2).astype(int)
    y_test_01 = ((y_test + 1) / 2).astype(int)

    if tracker:
        tracker.write('x_train', x_train.size)
        tracker.write('y_train', len(y_train))

    start = time.time()
    model.fit(x_train, y_train_01)
    train_time = time.time() - start

    if tracker:
        tracker.read('x_train')
        tracker.read('y_train')

    # Predictions
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    train_acc = float(np.mean(pred_train == y_train_01))
    test_acc = float(np.mean(pred_test == y_test_01))

    # Feature importances (which bits does the tree use?)
    importances = model.feature_importances_
    top_features = sorted(range(len(importances)),
                          key=lambda i: importances[i], reverse=True)[:10]
    top_importances = {int(i): round(float(importances[i]), 4) for i in top_features
                       if importances[i] > 0.01}

    return {
        'model': model_name,
        'train_acc': round(train_acc, 4),
        'test_acc': round(test_acc, 4),
        'train_time_s': round(train_time, 6),
        'top_features': top_importances,
    }


# =============================================================================
# RUN ONE CONFIG
# =============================================================================

def run_config(n_bits, k_sparse, n_train, seeds, verbose=True):
    """Run all models on one config across multiple seeds."""
    if verbose:
        print(f"\n  Config: n={n_bits}, k={k_sparse}, n_train={n_train}")
        print(f"  Seeds: {seeds}")

    all_model_results = {}

    # Define models to test
    model_specs = [
        ('DT_depth_k', lambda k: DecisionTreeClassifier(max_depth=k, random_state=0)),
        ('DT_depth_2k', lambda k: DecisionTreeClassifier(max_depth=2*k, random_state=0)),
        ('DT_unlimited', lambda k: DecisionTreeClassifier(max_depth=None, random_state=0)),
        ('RF_depth_k', lambda k: RandomForestClassifier(
            max_depth=k, n_estimators=100, random_state=0, n_jobs=-1)),
        ('RF_depth_2k', lambda k: RandomForestClassifier(
            max_depth=2*k, n_estimators=100, random_state=0, n_jobs=-1)),
        ('RF_depth_unlimited', lambda k: RandomForestClassifier(
            max_depth=None, n_estimators=100, random_state=0, n_jobs=-1)),
        ('ET_depth_k', lambda k: ExtraTreesClassifier(
            max_depth=k, n_estimators=500, random_state=0, n_jobs=-1)),
        ('ET_depth_2k', lambda k: ExtraTreesClassifier(
            max_depth=2*k, n_estimators=500, random_state=0, n_jobs=-1)),
        ('ET_depth_unlimited', lambda k: ExtraTreesClassifier(
            max_depth=None, n_estimators=500, random_state=0, n_jobs=-1)),
    ]

    for model_name, model_fn in model_specs:
        all_model_results[model_name] = []

    for seed in seeds:
        x_train, y_train, x_test, y_test, secret = generate_data(
            n_bits, k_sparse, n_train, seed=seed
        )

        if verbose:
            print(f"\n    seed={seed}, secret={secret}")

        tracker = MemTracker() if seed == seeds[0] else None

        for model_name, model_fn in model_specs:
            model = model_fn(k_sparse)
            result = run_model(model_name, model, x_train, y_train,
                               x_test, y_test, tracker=tracker)
            result['seed'] = seed
            result['secret'] = secret
            all_model_results[model_name].append(result)

            if verbose:
                # Check if top features contain the secret bits
                found_bits = set(result['top_features'].keys())
                secret_set = set(secret)
                overlap = found_bits & secret_set
                print(f"      {model_name:<22} train={result['train_acc']:.0%}  "
                      f"test={result['test_acc']:.0%}  "
                      f"time={result['train_time_s']:.4f}s  "
                      f"secret_overlap={len(overlap)}/{k_sparse}")

        if tracker:
            tracker.report()

    return {
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'n_train': n_train,
        'models': all_model_results,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: Decision Trees / Random Forests for Sparse Parity")
    print("  Approach #3: Can tree-based methods solve parity?")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # -------------------------------------------------------------------
    # Config 1: n=20, k=3
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=20, k=3")
    print("=" * 70)
    all_results['n20_k3'] = run_config(
        n_bits=20, k_sparse=3, n_train=5000, seeds=seeds
    )

    # -------------------------------------------------------------------
    # Config 2: n=50, k=3
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=50, k=3")
    print("=" * 70)
    all_results['n50_k3'] = run_config(
        n_bits=50, k_sparse=3, n_train=5000, seeds=seeds
    )

    # -------------------------------------------------------------------
    # Config 3: n=20, k=5
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 3: n=20, k=5")
    print("=" * 70)
    all_results['n20_k5'] = run_config(
        n_bits=20, k_sparse=5, n_train=10000, seeds=seeds
    )

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("  SUMMARY TABLE: Average test accuracy across seeds")
    print("=" * 100)

    model_names = ['DT_depth_k', 'DT_depth_2k', 'DT_unlimited',
                   'RF_depth_k', 'RF_depth_2k', 'RF_depth_unlimited',
                   'ET_depth_k', 'ET_depth_2k', 'ET_depth_unlimited']

    header = f"  {'Model':<22}"
    for config_key in ['n20_k3', 'n50_k3', 'n20_k5']:
        header += f" | {config_key:>12}"
    print(header)
    print("  " + "-" * 65)

    for model_name in model_names:
        row = f"  {model_name:<22}"
        for config_key in ['n20_k3', 'n50_k3', 'n20_k5']:
            results = all_results[config_key]['models'][model_name]
            avg_test = np.mean([r['test_acc'] for r in results])
            avg_train = np.mean([r['train_acc'] for r in results])
            row += f" | {avg_test:>5.1%} ({avg_train:.0%}tr)"
            # Store summary stats
            for r in results:
                r['avg_test_acc'] = round(avg_test, 4)
        print(row)

    # -------------------------------------------------------------------
    # Timing table
    # -------------------------------------------------------------------
    print("\n  TIMING (average seconds):")
    header = f"  {'Model':<22}"
    for config_key in ['n20_k3', 'n50_k3', 'n20_k5']:
        header += f" | {config_key:>10}"
    print(header)
    print("  " + "-" * 55)

    for model_name in model_names:
        row = f"  {model_name:<22}"
        for config_key in ['n20_k3', 'n50_k3', 'n20_k5']:
            results = all_results[config_key]['models'][model_name]
            avg_time = np.mean([r['train_time_s'] for r in results])
            row += f" | {avg_time:>9.4f}s"
        print(row)

    # -------------------------------------------------------------------
    # Comparison with baselines
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  COMPARISON WITH BASELINES")
    print("=" * 100)
    print(f"  {'Method':<30} | {'n=20/k=3':>12} | {'n=50/k=3':>12} | {'n=20/k=5':>12}")
    print("  " + "-" * 75)

    # Best tree result per config
    for config_key in ['n20_k3', 'n50_k3', 'n20_k5']:
        best_name = None
        best_acc = 0
        for model_name in model_names:
            results = all_results[config_key]['models'][model_name]
            avg = np.mean([r['test_acc'] for r in results])
            if avg > best_acc:
                best_acc = avg
                best_name = model_name
        all_results[config_key]['best_model'] = best_name
        all_results[config_key]['best_acc'] = round(best_acc, 4)

    # Print best tree per config
    row = f"  {'Best tree (this exp)':<30}"
    for config_key in ['n20_k3', 'n50_k3', 'n20_k5']:
        acc = all_results[config_key]['best_acc']
        name = all_results[config_key]['best_model']
        row += f" | {acc:>5.1%} ({name[:6]})"
    print(row)

    # SGD baselines
    print(f"  {'SGD (baseline)':<30} | {'100%':>12} | {'54% (FAIL)':>12} | {'100%':>12}")
    print(f"  {'Random search':<30} | {'100%':>12} | {'100%':>12} | {'100%':>12}")
    print(f"  {'Fourier exhaustive':<30} | {'100%':>12} | {'100%':>12} | {'100%':>12}")

    print("=" * 100)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_decision_tree'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'

    # Build JSON-safe version
    save_data = {
        'experiment': 'exp_decision_tree',
        'description': 'Decision Trees / Random Forests / ExtraTrees for sparse parity',
        'hypothesis': ('A depth-k tree can learn k-parity if splits target secret bits, '
                        'but greedy info-gain splitting fails because individual bits '
                        'have zero correlation with the label.'),
        'approach': 'sklearn tree-based classifiers with varying depth',
        'configs': {},
    }

    for config_key, config_data in all_results.items():
        save_config = {
            'n_bits': config_data['n_bits'],
            'k_sparse': config_data['k_sparse'],
            'n_train': config_data['n_train'],
            'best_model': config_data.get('best_model'),
            'best_acc': config_data.get('best_acc'),
            'models': {},
        }
        for model_name, runs in config_data['models'].items():
            save_config['models'][model_name] = {
                'avg_test_acc': round(float(np.mean([r['test_acc'] for r in runs])), 4),
                'avg_train_acc': round(float(np.mean([r['train_acc'] for r in runs])), 4),
                'avg_time_s': round(float(np.mean([r['train_time_s'] for r in runs])), 6),
                'runs': runs,
            }
        save_data['configs'][config_key] = save_config

    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Experiment {ID}: {Title}

Hypothesis: {If we do X, then Y because Z}
Answers: {Which open question from DISCOVERIES.md}

Usage:
    PYTHONPATH=src python3 src/sparse_parity/experiments/{this_file}.py
"""

import time
import json
import math
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params, forward, forward_batch
from sparse_parity.tracker import MemTracker
from sparse_parity.metrics import hinge_loss, accuracy

# =============================================================================
# CONFIG
# =============================================================================

EXP_NAME = "exp_X_name"  # Change this
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / EXP_NAME

CONFIG = Config(
    n_bits=20,
    k_sparse=3,
    hidden=1000,
    lr=0.1,
    wd=0.01,
    max_epochs=200,
    n_train=500,
    n_test=200,
    seed=42,
)

# =============================================================================
# BASELINE (for comparison — copy from existing train.py or adapt)
# =============================================================================

def run_baseline(config):
    """Run standard backprop as baseline. Return dict with accuracy, ARD, etc."""
    from sparse_parity.train import train
    data = generate(config)
    x_train, y_train, x_test, y_test, secret = data
    W1, b1, W2, b2 = init_params(config)
    result = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config, tracker_step=0)
    return result

# =============================================================================
# EXPERIMENT (your new thing)
# =============================================================================

def run_experiment(config):
    """Run your experimental variant. Return dict with same keys as baseline."""
    # TODO: Implement your experiment here
    # Must return dict with at least:
    #   best_test_acc, total_steps, elapsed_s, tracker (MemTracker.to_json() or None)
    pass

# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'='*70}")
    print(f"  EXPERIMENT: {EXP_NAME}")
    print(f"  Config: n={CONFIG.n_bits}, k={CONFIG.k_sparse}, hidden={CONFIG.hidden}")
    print(f"{'='*70}")

    # --- Baseline ---
    print("\n  [Baseline] Standard backprop...")
    t0 = time.time()
    baseline = run_baseline(CONFIG)
    print(f"    Accuracy: {baseline['best_test_acc']:.0%} in {time.time()-t0:.1f}s")
    if baseline.get('tracker'):
        print(f"    ARD: {baseline['tracker']['weighted_ard']:,.0f}")

    # --- Experiment ---
    print(f"\n  [Experiment] {EXP_NAME}...")
    t0 = time.time()
    experiment = run_experiment(CONFIG)
    if experiment:
        print(f"    Accuracy: {experiment['best_test_acc']:.0%} in {time.time()-t0:.1f}s")
        if experiment.get('tracker'):
            print(f"    ARD: {experiment['tracker']['weighted_ard']:,.0f}")

    # --- Comparison ---
    print(f"\n  {'='*70}")
    print(f"  COMPARISON")
    print(f"  {'='*70}")
    print(f"  {'Method':<20} {'Accuracy':>10} {'ARD':>12}")
    print(f"  {'─'*20} {'─'*10} {'─'*12}")
    b_ard = baseline['tracker']['weighted_ard'] if baseline.get('tracker') else 'N/A'
    print(f"  {'baseline':<20} {baseline['best_test_acc']:>9.0%} {b_ard:>12,}")
    if experiment:
        e_ard = experiment['tracker']['weighted_ard'] if experiment.get('tracker') else 'N/A'
        print(f"  {'experiment':<20} {experiment['best_test_acc']:>9.0%} {e_ard:>12,}")

    # --- Save ---
    results = {
        'experiment': EXP_NAME,
        'config': {k: v for k, v in CONFIG.__dict__.items()},
        'baseline': {k: v for k, v in baseline.items() if k != 'tracker'},
        'baseline_ard': baseline.get('tracker'),
    }
    if experiment:
        results['experiment_result'] = {k: v for k, v in experiment.items() if k != 'tracker'}
        results['experiment_ard'] = experiment.get('tracker')

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_DIR / 'results.json'}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Experiment Cache-ARD: Cache-aware ARD comparison.

Hypothesis: With LRU cache simulation, batch-32 will show dramatically
higher hit rates and lower effective ARD than single-sample, especially
at L2 cache sizes where W1 fits.

Answers: Open question #2 from DISCOVERIES.md — "What does ARD look like
with a cache model?"

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_cache_ard.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params
from sparse_parity.cache_tracker import CacheTracker
from sparse_parity.metrics import save_json

# Reuse the instrumented step functions from exp_b
from sparse_parity.experiments.exp_b_batch_ard import single_sample_step, batch_step


# =============================================================================
# CONFIG
# =============================================================================

EXP_NAME = "exp_cache_ard"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / EXP_NAME

BATCH_SIZE = 32

# Two hidden sizes to show the crossover
# hidden=200: W1=15.6KB, fits in L1
# hidden=1000: W1=78KB, exceeds L1 but fits in L2
HIDDEN_SIZES = [200, 1000]

# Cache sizes to simulate
# L1: 32KB = 8,192 float32s
# L2: 256KB = 65,536 float32s
CACHE_CONFIGS = {
    'L1_32KB': 8192,
    'L2_256KB': 65536,
}


def make_config(hidden):
    return Config(
        n_bits=20,
        k_sparse=3,
        n_train=500,
        n_test=200,
        hidden=hidden,
        lr=0.1,
        wd=0.01,
        max_epochs=1,
        seed=42,
    )


# =============================================================================
# RUN
# =============================================================================

def run_single_sample(config, batch_size, cache_size_floats):
    """Run batch_size consecutive single-sample steps with cache tracking."""
    x_train, y_train, _, _, _ = generate(config)
    W1, b1, W2, b2 = init_params(config)

    tracker = CacheTracker(cache_size_floats)
    t0 = time.time()

    for i in range(batch_size):
        single_sample_step(
            x_train[i], y_train[i],
            W1, b1, W2, b2,
            config, tracker
        )

    elapsed = time.time() - t0
    return tracker, elapsed


def run_batch(config, batch_size, cache_size_floats):
    """Run one batch step with cache tracking."""
    x_train, y_train, _, _, _ = generate(config)
    W1, b1, W2, b2 = init_params(config)

    tracker = CacheTracker(cache_size_floats)
    t0 = time.time()

    batch_step(
        x_train[:batch_size], y_train[:batch_size],
        W1, b1, W2, b2,
        config, tracker
    )

    elapsed = time.time() - t0
    return tracker, elapsed


def run_comparison(config, cache_name, cache_size):
    """Run single-sample vs batch comparison for one config+cache pair."""
    tracker_ss, t_ss = run_single_sample(config, BATCH_SIZE, cache_size)
    ss_summary = tracker_ss.summary()
    ss_cache = tracker_ss.cache_summary()

    tracker_b, t_b = run_batch(config, BATCH_SIZE, cache_size)
    b_summary = tracker_b.summary()
    b_cache = tracker_b.cache_summary()

    return {
        'single_sample': {
            'weighted_ard': ss_summary['weighted_ard'],
            'total_floats': ss_summary['total_floats_accessed'],
            'reads': ss_summary['reads'],
            'writes': ss_summary['writes'],
            'cache_hits': ss_cache['hits'],
            'cache_misses': ss_cache['misses'],
            'hit_rate': ss_cache['hit_rate'],
            'effective_ard': ss_cache['effective_ard'],
            'wall_time_s': t_ss,
        },
        'batch_32': {
            'weighted_ard': b_summary['weighted_ard'],
            'total_floats': b_summary['total_floats_accessed'],
            'reads': b_summary['reads'],
            'writes': b_summary['writes'],
            'cache_hits': b_cache['hits'],
            'cache_misses': b_cache['misses'],
            'hit_rate': b_cache['hit_rate'],
            'effective_ard': b_cache['effective_ard'],
            'wall_time_s': t_b,
        },
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  EXPERIMENT: Cache-Aware ARD Comparison")
    print("=" * 70)
    print(f"  Hidden sizes: {HIDDEN_SIZES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Cache configs: {', '.join(f'{k} ({v:,} floats)' for k, v in CACHE_CONFIGS.items())}")
    print()

    all_results = {
        'experiment': EXP_NAME,
        'hidden_sizes': HIDDEN_SIZES,
        'batch_size': BATCH_SIZE,
        'cache_configs': {k: v for k, v in CACHE_CONFIGS.items()},
        'runs': {},
    }

    # Collect all rows for final summary table
    summary_rows = []

    for hidden in HIDDEN_SIZES:
        config = make_config(hidden)
        w1_floats = hidden * config.n_bits
        w1_kb = w1_floats * 4 / 1024

        print(f"\n{'=' * 70}")
        print(f"  HIDDEN={hidden}  (W1 = {w1_floats:,} floats = {w1_kb:.1f} KB)")
        print(f"{'=' * 70}")

        run_key = f"hidden_{hidden}"
        all_results['runs'][run_key] = {
            'hidden': hidden,
            'w1_floats': w1_floats,
            'w1_kb': w1_kb,
            'comparisons': {},
        }

        for cache_name, cache_size in CACHE_CONFIGS.items():
            cache_kb = cache_size * 4 / 1024
            w1_fits = w1_floats <= cache_size

            print(f"\n  {'─' * 66}")
            print(f"  CACHE: {cache_name} ({cache_kb:.0f} KB) — "
                  f"W1 fits? {'YES' if w1_fits else 'NO'}")
            print(f"  {'─' * 66}")

            comp = run_comparison(config, cache_name, cache_size)
            ss = comp['single_sample']
            b = comp['batch_32']

            print(f"\n  {'Metric':<30} {'Single-Sample':>15} {'Batch-32':>15}")
            print(f"  {'─'*30} {'─'*15} {'─'*15}")
            print(f"  {'Cache hits':<30} {ss['cache_hits']:>15,} {b['cache_hits']:>15,}")
            print(f"  {'Cache misses':<30} {ss['cache_misses']:>15,} {b['cache_misses']:>15,}")
            print(f"  {'Hit rate':<30} {ss['hit_rate']:>14.1%} {b['hit_rate']:>14.1%}")
            print(f"  {'Effective ARD (misses)':<30} {ss['effective_ard']:>15,.0f} {b['effective_ard']:>15,.0f}")
            print(f"  {'Raw ARD':<30} {ss['weighted_ard']:>15,.0f} {b['weighted_ard']:>15,.0f}")
            print(f"  {'Total floats accessed':<30} {ss['total_floats']:>15,} {b['total_floats']:>15,}")

            all_results['runs'][run_key]['comparisons'][cache_name] = {
                'cache_size_floats': cache_size,
                'w1_fits_in_cache': w1_fits,
                **comp,
            }

            summary_rows.append((hidden, cache_name, w1_fits, ss, b))

    # --- Grand summary ---
    print(f"\n{'=' * 70}")
    print(f"  GRAND SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Hidden':>6} {'Cache':<12} {'W1 fits':>7} {'Method':<14} "
          f"{'Hit%':>6} {'Eff.ARD':>10} {'Misses':>8}")
    print(f"  {'─'*6} {'─'*12} {'─'*7} {'─'*14} {'─'*6} {'─'*10} {'─'*8}")

    for hidden, cache_name, w1_fits, ss, b in summary_rows:
        fit_str = 'YES' if w1_fits else 'NO'
        print(f"  {hidden:>6} {cache_name:<12} {fit_str:>7} {'single-sample':<14} "
              f"{ss['hit_rate']:>5.0%} {ss['effective_ard']:>10,.0f} {ss['cache_misses']:>8,}")
        print(f"  {'':>6} {'':12} {'':>7} {'batch-32':<14} "
              f"{b['hit_rate']:>5.0%} {b['effective_ard']:>10,.0f} {b['cache_misses']:>8,}")

    print(f"{'=' * 70}")

    # --- Save ---
    results_path = RESULTS_DIR / 'results.json'
    save_json(all_results, results_path)
    print(f"\n  Results saved to: {results_path}")

    return all_results


if __name__ == '__main__':
    main()

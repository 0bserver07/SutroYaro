#!/usr/bin/env python3
"""
Experiment: Reinforcement Learning Bit Querying for Sparse Parity

Hypothesis: An RL agent can learn which k bits to query by observing
rewards for correct/incorrect parity predictions. The optimal policy
queries exactly the k secret bits. This reframes energy-efficient
learning as "learning what to look at" with minimum memory reads.

Two approaches:
  1. Bandit over k-subsets: each arm is a k-subset, UCB1 selection,
     reward = accuracy on a mini-batch. Simple and tractable.
  2. Sequential bit-querying agent: at each step pick a bit to query,
     after k queries predict label. Tabular Q-learning with
     epsilon-greedy exploration.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_rl.py
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
# APPROACH 1: BANDIT OVER K-SUBSETS (UCB1)
# =============================================================================

def evaluate_subset(x, y, subset, batch_indices):
    """Evaluate accuracy of a k-subset on a batch of samples."""
    parity = np.prod(x[np.ix_(batch_indices, list(subset))], axis=1)
    return np.mean(parity == y[batch_indices])


def bandit_ucb_search(x, y, n_bits, k_sparse, n_episodes=2000,
                      batch_size=50, seed=42):
    """
    Bandit over k-subsets with UCB1.

    Each arm = a k-subset of {0..n-1}.
    Pull = evaluate accuracy on a random mini-batch.
    Reward = fraction correct.

    For small C(n,k), enumerate all arms. For large C(n,k), use a
    random sample of arms plus progressive widening.

    Returns dict with results.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y)
    c_n_k = comb(n_bits, k_sparse)

    # Enumerate or sample arms
    if c_n_k <= 5000:
        arms = list(combinations(range(n_bits), k_sparse))
    else:
        # Sample a manageable number of arms, ensure we include enough diversity
        arm_set = set()
        while len(arm_set) < min(5000, c_n_k):
            subset = tuple(sorted(rng.choice(n_bits, k_sparse, replace=False).tolist()))
            arm_set.add(subset)
        arms = list(arm_set)

    n_arms = len(arms)
    counts = np.zeros(n_arms)       # pull counts
    total_reward = np.zeros(n_arms)  # sum of rewards
    best_arm_history = []

    # Memory tracker
    tracker = MemTracker()

    start = time.time()

    # Phase 1: pull each arm once
    for i in range(n_arms):
        batch_idx = rng.choice(n_samples, batch_size, replace=False)
        # Track memory: read k bits per sample in batch
        tracker.write(f'arm_{i}_query', k_sparse * batch_size)
        reward = evaluate_subset(x, y, arms[i], batch_idx)
        tracker.read(f'arm_{i}_query', k_sparse * batch_size)
        counts[i] = 1
        total_reward[i] = reward

    found_at_episode = None

    # Phase 2: UCB1 selection
    for ep in range(n_arms, n_episodes):
        # UCB1 scores
        means = total_reward / counts
        ucb_bonus = np.sqrt(2 * np.log(ep + 1) / counts)
        ucb_scores = means + ucb_bonus

        arm_idx = np.argmax(ucb_scores)
        arm = arms[arm_idx]

        batch_idx = rng.choice(n_samples, batch_size, replace=False)
        tracker.write(f'arm_{arm_idx}_query', k_sparse * batch_size)
        reward = evaluate_subset(x, y, arm, batch_idx)
        tracker.read(f'arm_{i}_query', k_sparse * batch_size)

        counts[arm_idx] += 1
        total_reward[arm_idx] += reward

        # Track best arm
        best_idx = np.argmax(total_reward / counts)
        best_arm_history.append(list(arms[best_idx]))

        # Check if best arm is perfect
        if found_at_episode is None and (total_reward[best_idx] / counts[best_idx]) >= 0.99:
            # Verify on full dataset
            full_acc = evaluate_subset(x, y, arms[best_idx], np.arange(n_samples))
            if full_acc >= 1.0:
                found_at_episode = ep

    elapsed = time.time() - start

    # Final best arm
    means = total_reward / counts
    best_idx = np.argmax(means)
    best_arm = list(arms[best_idx])
    best_mean = float(means[best_idx])

    # Final verification
    full_acc = evaluate_subset(x, y, arms[best_idx], np.arange(n_samples))

    return {
        'method': 'bandit_ucb',
        'best_arm': best_arm,
        'best_mean_reward': round(best_mean, 4),
        'full_accuracy': float(full_acc),
        'solved': full_acc >= 1.0,
        'found_at_episode': found_at_episode,
        'total_episodes': n_episodes,
        'n_arms': n_arms,
        'elapsed_s': round(elapsed, 4),
        'memory': tracker.to_json(),
    }


# =============================================================================
# APPROACH 2: SEQUENTIAL BIT-QUERYING AGENT (Q-LEARNING)
# =============================================================================
#
# Key insight: the optimal policy for "which bit to query next" does NOT
# depend on the values of previously queried bits -- it only depends on
# WHICH bits have been queried. The prediction step (product of values)
# is deterministic once the subset is chosen. So we use a value-blind
# state: just the frozenset of queried bit indices.
#
# State space: sum_{j=0}^{k-1} C(n, j) possible states (much smaller).
# For n=10, k=3: 1 + 10 + 45 = 56 states.

def sequential_agent(x, y, n_bits, k_sparse, n_episodes=5000,
                     alpha=0.2, gamma=0.99, epsilon_start=1.0,
                     epsilon_end=0.02, epsilon_decay=0.9995,
                     seed=42):
    """
    Sequential bit-querying agent using tabular Q-learning.

    State: frozenset of queried bit indices (value-blind)
    Action: which bit index to query next (from unqueried bits)
    After k queries: predict label = product of queried bit values
    Reward: +1 correct, -1 wrong (only at terminal step)

    The value-blind state makes the Q-table tractable.
    The agent learns WHICH bits to query, not what values to expect.

    Returns dict with results.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y)
    Q = {}  # Q[(frozenset_of_queried_bits, action)] -> value
    tracker = MemTracker()

    epsilon = epsilon_start
    start = time.time()

    episode_rewards = []
    window_size = 500
    convergence_episode = None

    for ep in range(n_episodes):
        # Sample a random data point
        idx = rng.randint(n_samples)
        sample_x = x[idx]
        sample_y = y[idx]

        queried_set = frozenset()
        queried_values = {}  # bit_index -> value

        transitions = []  # store (state, action) for backward update

        for step in range(k_sparse):
            state = queried_set

            # Available actions: bits not yet queried
            available = [b for b in range(n_bits) if b not in queried_set]

            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = available[rng.randint(len(available))]
            else:
                q_values = [Q.get((state, a), 0.0) for a in available]
                best_q = max(q_values)
                best_actions = [a for a, q in zip(available, q_values) if q == best_q]
                action = best_actions[rng.randint(len(best_actions))]

            # Execute action: query the bit
            bit_value = sample_x[action]
            tracker.write(f'query_bit_{action}', 1)
            tracker.read(f'query_bit_{action}', 1)

            transitions.append((state, action))
            queried_set = queried_set | {action}
            queried_values[action] = bit_value

        # Compute reward: predict using product of queried values
        prediction = 1.0
        for v in queried_values.values():
            prediction *= v
        reward = 1.0 if prediction == sample_y else -1.0

        # Backward Q-learning update (reward only at terminal step)
        # Last step: terminal
        s_last, a_last = transitions[-1]
        old_q = Q.get((s_last, a_last), 0.0)
        Q[(s_last, a_last)] = old_q + alpha * (reward - old_q)

        # Earlier steps: propagate with gamma
        for i in range(len(transitions) - 2, -1, -1):
            s_i, a_i = transitions[i]
            s_next = transitions[i + 1][0] | {transitions[i][1]}  # state after action
            available_next = [b for b in range(n_bits) if b not in s_next]
            if available_next:
                max_q_next = max(Q.get((s_next, a), 0.0) for a in available_next)
            else:
                max_q_next = 0.0
            old_q = Q.get((s_i, a_i), 0.0)
            Q[(s_i, a_i)] = old_q + alpha * (0 + gamma * max_q_next - old_q)

        episode_rewards.append(reward)

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Check convergence
        if convergence_episode is None and ep >= window_size:
            recent = episode_rewards[-window_size:]
            recent_acc = sum(1 for r in recent if r > 0) / window_size
            if recent_acc >= 0.98:
                convergence_episode = ep

    elapsed = time.time() - start

    # Extract learned policy: greedily pick best bits from empty state
    learned_bits = []
    queried_set = frozenset()
    for step in range(k_sparse):
        available = [b for b in range(n_bits) if b not in queried_set]
        q_values = [Q.get((queried_set, a), 0.0) for a in available]
        if q_values:
            best_q = max(q_values)
            best_actions = [a for a, q in zip(available, q_values) if q == best_q]
            action = best_actions[0]
        else:
            action = available[0]
        learned_bits.append(action)
        queried_set = queried_set | {action}

    learned_bits_sorted = sorted(learned_bits)

    # Evaluate final policy accuracy on test data
    test_correct = 0
    n_test = min(1000, n_samples)
    test_indices = rng.choice(n_samples, n_test, replace=False)

    for t_idx in test_indices:
        sample_x = x[t_idx]
        sample_y = y[t_idx]
        queried_set = frozenset()
        queried_values = {}

        for step in range(k_sparse):
            available = [b for b in range(n_bits) if b not in queried_set]
            q_values = [Q.get((queried_set, a), 0.0) for a in available]
            if q_values:
                best_q = max(q_values)
                best_actions = [a for a, q in zip(available, q_values) if q == best_q]
                action = best_actions[0]
            else:
                action = available[0]

            queried_values[action] = sample_x[action]
            queried_set = queried_set | {action}

        prediction = 1.0
        for v in queried_values.values():
            prediction *= v
        if prediction == sample_y:
            test_correct += 1

    test_acc = test_correct / n_test

    # Rolling accuracy at end
    final_window = episode_rewards[-window_size:] if len(episode_rewards) >= window_size else episode_rewards
    final_rolling_acc = sum(1 for r in final_window if r > 0) / len(final_window)

    return {
        'method': 'sequential_qlearning',
        'learned_bits': learned_bits_sorted,
        'test_accuracy': round(test_acc, 4),
        'final_rolling_accuracy': round(final_rolling_acc, 4),
        'solved': test_acc >= 0.99,
        'convergence_episode': convergence_episode,
        'total_episodes': n_episodes,
        'q_table_size': len(Q),
        'elapsed_s': round(elapsed, 4),
        'memory': tracker.to_json(),
    }


# =============================================================================
# MAIN
# =============================================================================

def run_config(n_bits, k_sparse, n_train, seeds, bandit_episodes=2000,
               seq_episodes=10000, verbose=True):
    """Run both RL approaches on one config across seeds."""
    c_n_k = comb(n_bits, k_sparse)
    if verbose:
        print(f"\n  Config: n={n_bits}, k={k_sparse}, C(n,k)={c_n_k}")
        print(f"  Training samples: {n_train}")

    bandit_results = []
    seq_results = []

    for seed in seeds:
        x, y, secret = generate_data(n_bits, k_sparse, n_train, seed=seed)

        # --- Bandit UCB ---
        if verbose:
            print(f"    [Bandit UCB] seed={seed} ... ", end='', flush=True)
        res_b = bandit_ucb_search(
            x, y, n_bits, k_sparse,
            n_episodes=bandit_episodes, batch_size=50, seed=seed + 100
        )
        res_b['secret'] = secret
        res_b['correct'] = sorted(res_b['best_arm']) == secret
        bandit_results.append(res_b)
        if verbose:
            status = "SOLVED" if res_b['solved'] else f"FAILED ({res_b['full_accuracy']:.1%})"
            ep_info = f" at ep {res_b['found_at_episode']}" if res_b['found_at_episode'] else ""
            print(f"{status}{ep_info} ({res_b['elapsed_s']:.3f}s)")

        # --- Sequential Q-learning ---
        # Value-blind state space: sum_{j=0}^{k-1} C(n,j) states
        # n=10,k=3: 56 states; n=20,k=3: 211 states -- both tractable
        if n_bits <= 20:
            if verbose:
                print(f"    [Seq QL]     seed={seed} ... ", end='', flush=True)
            res_s = sequential_agent(
                x, y, n_bits, k_sparse,
                n_episodes=seq_episodes,
                alpha=0.1, gamma=0.99,
                epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999,
                seed=seed + 200
            )
            res_s['secret'] = secret
            res_s['correct'] = sorted(res_s['learned_bits']) == secret
            seq_results.append(res_s)
            if verbose:
                status = "SOLVED" if res_s['solved'] else f"FAILED ({res_s['test_accuracy']:.1%})"
                conv = f" conv@{res_s['convergence_episode']}" if res_s['convergence_episode'] else ""
                print(f"{status}{conv} ({res_s['elapsed_s']:.3f}s) bits={res_s['learned_bits']}")
        else:
            if verbose:
                print(f"    [Seq QL]     seed={seed} ... SKIPPED (n={n_bits} too large)")
            seq_results.append({
                'method': 'sequential_qlearning',
                'skipped': True,
                'reason': f'n={n_bits} too large for tabular Q-learning',
                'secret': secret,
            })

    return {
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'c_n_k': c_n_k,
        'n_train': n_train,
        'bandit_ucb': bandit_results,
        'sequential_qlearning': seq_results,
    }


def main():
    print("=" * 70)
    print("  EXPERIMENT: RL Bit Querying for Sparse Parity")
    print("  Approach #16: Learning what to look at")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # -------------------------------------------------------------------
    # Config 1: n=10, k=3 — C(10,3) = 120, tractable for both methods
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=10, k=3  [C(10,3) = 120]")
    print("=" * 70)
    all_results['n10_k3'] = run_config(
        n_bits=10, k_sparse=3, n_train=500, seeds=seeds,
        bandit_episodes=1000, seq_episodes=20000
    )

    # -------------------------------------------------------------------
    # Config 2: n=20, k=3 — C(20,3) = 1140
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=20, k=3  [C(20,3) = 1,140]")
    print("=" * 70)
    all_results['n20_k3'] = run_config(
        n_bits=20, k_sparse=3, n_train=500, seeds=seeds,
        bandit_episodes=3000, seq_episodes=50000
    )

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)

    header = (f"  {'Config':<12} | {'C(n,k)':>8} | "
              f"{'Bandit solved':>14} | {'Bandit ep':>10} | {'Bandit time':>11} | "
              f"{'SeqQL solved':>12} | {'SeqQL conv':>10} | {'SeqQL time':>11}")
    print(header)
    print("  " + "-" * 100)

    for key, res in all_results.items():
        # Bandit stats
        b_solved = sum(1 for r in res['bandit_ucb'] if r.get('solved', False))
        b_eps = [r['found_at_episode'] for r in res['bandit_ucb']
                 if r.get('found_at_episode') is not None]
        b_ep_avg = np.mean(b_eps) if b_eps else float('nan')
        b_time = np.mean([r['elapsed_s'] for r in res['bandit_ucb']
                          if not r.get('skipped', False)])

        # Sequential stats
        s_not_skipped = [r for r in res['sequential_qlearning'] if not r.get('skipped', False)]
        s_solved = sum(1 for r in s_not_skipped if r.get('solved', False))
        s_convs = [r['convergence_episode'] for r in s_not_skipped
                   if r.get('convergence_episode') is not None]
        s_conv_avg = np.mean(s_convs) if s_convs else float('nan')
        s_time = np.mean([r['elapsed_s'] for r in s_not_skipped]) if s_not_skipped else float('nan')

        n_total_b = len(res['bandit_ucb'])
        n_total_s = len(s_not_skipped) if s_not_skipped else 0

        b_ep_str = f"{b_ep_avg:.0f}" if not np.isnan(b_ep_avg) else "---"
        b_t_str = f"{b_time:.3f}s"
        s_str = f"{s_solved}/{n_total_s}" if n_total_s > 0 else "SKIP"
        s_conv_str = f"{s_conv_avg:.0f}" if not np.isnan(s_conv_avg) else "---"
        s_t_str = f"{s_time:.3f}s" if not np.isnan(s_time) else "---"

        print(f"  {key:<12} | {res['c_n_k']:>8,} | "
              f"{b_solved}/{n_total_b:>13} | {b_ep_str:>10} | {b_t_str:>11} | "
              f"{s_str:>12} | {s_conv_str:>10} | {s_t_str:>11}")

    # -------------------------------------------------------------------
    # Memory efficiency analysis
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  MEMORY EFFICIENCY (ARD ANALYSIS)")
    print("=" * 70)

    for key, res in all_results.items():
        print(f"\n  Config: {key}")
        # Bandit memory
        b_mem = res['bandit_ucb'][0].get('memory', {})
        if b_mem:
            print(f"    Bandit UCB ARD: {b_mem.get('weighted_ard', 0):,.0f} floats")
            print(f"    Bandit reads: {b_mem.get('reads', 0):,}, writes: {b_mem.get('writes', 0):,}")

        # Sequential memory
        s_first = [r for r in res['sequential_qlearning'] if not r.get('skipped', False)]
        if s_first:
            s_mem = s_first[0].get('memory', {})
            if s_mem:
                print(f"    Seq QL ARD: {s_mem.get('weighted_ard', 0):,.0f} floats")
                print(f"    Seq reads: {s_mem.get('reads', 0):,}, writes: {s_mem.get('writes', 0):,}")
        print(f"    Optimal reads per prediction: k={res['k_sparse']} bits")

    print("\n" + "=" * 70)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_rl'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Make results JSON serializable
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(clean_for_json({
            'experiment': 'exp_rl',
            'description': 'RL bit querying for sparse parity — bandit and sequential Q-learning',
            'hypothesis': 'RL agent learns to query exactly the k secret bits, minimizing memory reads',
            'approach': 'Reinforcement Learning — Bit Querying (#16)',
            'configs': all_results,
        }), f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

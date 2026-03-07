# Experiment exp_rl: Reinforcement Learning Bit Querying for Sparse Parity

**Date**: 2026-03-06
**Status**: SUCCESS
**Answers**: Can an RL agent learn which k bits to query, reframing sparse parity as "learning what to look at"?

## Hypothesis

An RL agent can learn to query exactly the k secret bits by observing rewards for correct/incorrect parity predictions. Two formulations:
1. **Bandit over k-subsets (UCB1)**: each arm is a k-subset, pull = evaluate accuracy on a batch, UCB1 for exploration. Should find the correct arm after ~C(n,k) pulls (one per arm).
2. **Sequential bit-querying (Q-learning)**: at each step pick a bit to query, after k queries predict label = product of values. The agent must learn the optimal query policy through trial and error.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 10, 20 |
| k_sparse | 3 |
| n_train | 500 |
| seeds | 42, 43, 44 |
| Bandit episodes | 1000 (n=10), 3000 (n=20) |
| Seq QL episodes | 20000 (n=10), 50000 (n=20) |
| Seq QL alpha | 0.2 |
| Seq QL epsilon | 1.0 -> 0.02 (decay 0.9995) |

## Results

| Config | C(n,k) | Bandit: found at ep | Bandit: time | SeqQL: converged at ep | SeqQL: time | Both correct |
|--------|--------|---------------------|--------------|------------------------|-------------|-------------|
| n=10, k=3 | 120 | 120 (all seeds) | 0.034s avg | 3656, 4169, 4377 | 0.61s avg | 6/6 |
| n=20, k=3 | 1,140 | 1140 (all seeds) | 0.12s avg | 9526, 11907, 5122 | 2.40s avg | 6/6 |

### Memory Efficiency

| Config | Method | ARD (floats) | Reads | Writes | Reads per prediction |
|--------|--------|--------------|-------|--------|---------------------|
| n=10, k=3 | Bandit UCB | 16,022 | 1,000 | 1,000 | k*batch = 150 |
| n=10, k=3 | Seq QL | 1 | 60,000 | 60,000 | k = 3 |
| n=20, k=3 | Bandit UCB | 83,396 | 3,000 | 3,000 | k*batch = 150 |
| n=20, k=3 | Seq QL | 1 | 150,000 | 150,000 | k = 3 |

## Analysis

### What worked

- **Both approaches recover the exact secret bits in all 6 runs** (3 seeds x 2 configs), with 100% test accuracy.
- **Bandit UCB is reliable and fast**: finds the correct arm after exactly C(n,k) episodes (one per arm in the initialization sweep). The UCB exploration bonus then confirms the best arm. Wall time is 0.03-0.12s.
- **Sequential Q-learning succeeds with value-blind state**: the critical design decision is using only the set of queried bit indices as state (not their values). This reduces the state space from exponential to sum_{j=0}^{k-1} C(n,j), which is 56 for n=10/k=3 and 211 for n=20/k=3.
- **Sequential agent achieves minimal ARD**: each prediction reads exactly k=3 bits. The ARD of 1 float means each bit read is immediately used (no cache miss). This is the theoretical minimum for energy-efficient parity computation.

### What didn't work

- **Value-aware Q-learning fails completely**: the initial implementation tracked (queried_bits, queried_values) as state, giving ~50% accuracy (random chance). The state space explodes to O(n^k * 2^k) and the Q-table never converges.
- **Sequential Q-learning is slower than bandit**: 4000-12000 episodes to converge vs. ~C(n,k) for bandit. The sequential agent must learn a policy over k steps, whereas the bandit directly evaluates complete subsets.
- **Bandit UCB has high ARD**: each arm pull evaluates a batch of 50 samples, reading k*50=150 floats per pull. This is necessary for accurate reward estimation but is much less memory-efficient than the sequential agent's 3 reads per prediction.

### Key insight: the information-theoretic view

The bandit finds the correct arm after exactly C(n,k) pulls because it must try each arm at least once. This matches the information-theoretic lower bound: distinguishing 1 correct subset from C(n,k)-1 wrong ones requires at least C(n,k) observations. The sequential agent is slower because it explores incrementally, but once converged, it achieves the optimal k reads per prediction.

### Comparison with other approaches

| Method | n=10/k=3 time | n=20/k=3 time | Reads per prediction |
|--------|--------------|--------------|---------------------|
| Bandit UCB | 0.034s | 0.12s | k*batch (150) |
| Sequential QL | 0.61s | 2.40s | k (3) -- optimal |
| Random search (exp_evolutionary) | ~0.01s | ~0.01s | n_train*k (1500) |
| SGD neural net | 0.12s | 0.12s | n (all bits) |

The sequential QL agent is the only approach that achieves the theoretical minimum of k reads per prediction at inference time. All other methods read more data per prediction.

## Open Questions

- Can the sequential agent scale to k=5 or larger? The value-blind state space grows as sum C(n,j) for j < k, which is still polynomial for small k.
- A neural network policy (DQN) could replace the Q-table and potentially handle larger n by generalizing across states.
- Hybrid: use bandit to identify candidate subsets, then train a sequential agent restricted to top candidates.

## Files

- Experiment: `src/sparse_parity/experiments/exp_rl.py`
- Results: `results/exp_rl/results.json`

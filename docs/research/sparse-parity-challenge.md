# Sparse Parity Challenge

Submit a Python function that solves sparse parity. Your solution is automatically evaluated for accuracy, speed, and energy cost (DMC).

[![Submit Solution](https://img.shields.io/badge/Submit-Solution-blue?style=for-the-badge)](https://github.com/cybertronai/sparse-parity-challenge/issues/new?template=submission.yml)

## The problem

Given `x` (n random {-1,+1} inputs) and `y` (product of k secret bits), find the k secret bit indices. Default: n=20 bits, k=3 secret, 17 noise.

## How it works

1. Click "Submit Solution" above (opens a GitHub Issue form)
2. Paste your `solve()` function
3. GitHub Actions runs it with TrackedArray (auto-measures memory access)
4. Bot posts results: accuracy, DMC, wall time
5. If accuracy >= 95%: a PR is opened to update the leaderboard
6. Maintainer merges, leaderboard updates

## Function signature

```python
def solve(x, y, n_bits, k_sparse):
    """
    Args:
        x: numpy array (n_samples, n_bits), values in {-1, +1}
        y: numpy array (n_samples,), values in {-1, +1}
        n_bits: int, total number of bits
        k_sparse: int, number of secret bits
    Returns:
        list[int]: sorted indices of the k secret bits
    """
```

Only `numpy` allowed. 60-second timeout. Runs across 3 random seeds.

## What we measure

**DMC (Data Movement Complexity)**: how much energy your solution costs. Every numpy operation on your arrays is automatically tracked via TrackedArray. Each array read has a cost based on its reuse distance (how long ago the data was last written). Lower DMC = data stays in cache = less energy.

The neural network baseline (SGD) has DMC ~1.3M. The best algebraic solver (GF(2)) has DMC ~13.5M but solves in 509 microseconds. Can you find a better tradeoff?

## Leaderboard

See the [live leaderboard](https://github.com/cybertronai/sparse-parity-challenge#leaderboard) on the challenge repo.

## What doesn't work (save yourself time)

From 36 experiments in the [research repo](https://github.com/cybertronai/SutroYaro):

- **First-order correlations**: `E[x_i * y] = 0` for ALL bits, even secret ones. Parity has zero first-order signal.
- **Pairwise correlations**: `E[x_i * x_j * y] = 0` for ALL pairs. You must test the full k-way interaction.
- **Local learning rules**: Hebbian, Predictive Coding, Equilibrium Propagation, Target Propagation all fail at chance level.
- **Greedy feature selection**: forward selection fails because there's no single bit that's informative alone.

What works: algebraic methods (GF(2), KM influence), exhaustive search (Fourier), constraint satisfaction (SMT), and neural networks (SGD, but slow).

## Links

- [Challenge repo](https://github.com/cybertronai/sparse-parity-challenge) -- submissions, leaderboard, evaluation code
- [Research repo](https://github.com/cybertronai/SutroYaro) -- 36 experiments, findings, TrackedArray source
- [Practitioner's Field Guide](survey.md) -- ranked comparison of all methods
- [TrackedArray docs](tracked-numpy.md) -- how DMC measurement works

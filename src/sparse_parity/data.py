"""Phase 1: Dataset generation for sparse parity."""

import random

from .config import Config


def generate(config: Config):
    """
    Generate (n,k)-sparse parity train/test datasets.

    Returns (x_train, y_train, x_test, y_test, secret_indices).
    Inputs are {-1, +1}. Labels are product of inputs at secret indices.
    """
    rng = random.Random(config.seed)

    # Pick secret parity indices
    secret = sorted(rng.sample(range(config.n_bits), config.k_sparse))

    def make_data(n):
        xs, ys = [], []
        for _ in range(n):
            x = [rng.choice([-1.0, 1.0]) for _ in range(config.n_bits)]
            y = 1.0
            for idx in secret:
                y *= x[idx]
            xs.append(x)
            ys.append(y)
        return xs, ys

    x_train, y_train = make_data(config.n_train)
    x_test, y_test = make_data(config.n_test)

    return x_train, y_train, x_test, y_test, secret

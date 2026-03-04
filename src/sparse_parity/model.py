"""Phase 2: MLP model — init and forward pass."""

import math
import random

from .config import Config


def init_params(config: Config):
    """Initialize 2-layer MLP: input -> hidden (ReLU) -> scalar. Kaiming init."""
    rng = random.Random(config.seed + 1)  # different seed from data
    std1 = math.sqrt(2.0 / config.n_bits)
    std2 = math.sqrt(2.0 / config.hidden)

    W1 = [[rng.gauss(0, std1) for _ in range(config.n_bits)] for _ in range(config.hidden)]
    b1 = [0.0] * config.hidden
    W2 = [[rng.gauss(0, std2) for _ in range(config.hidden)]]
    b2 = [0.0]

    return W1, b1, W2, b2


def forward(x, W1, b1, W2, b2, tracker=None):
    """
    Forward pass for a single sample.
    x -> W1*x + b1 -> ReLU -> W2*h + b2 -> scalar
    Returns (out, h_pre, h).
    """
    hidden = len(W1)
    n_bits = len(x)

    if tracker:
        tracker.read('x', n_bits)
        tracker.read('W1', hidden * n_bits)
        tracker.read('b1', hidden)

    h_pre = [sum(W1[j][i] * x[i] for i in range(n_bits)) + b1[j] for j in range(hidden)]

    if tracker:
        tracker.write('h_pre', hidden)
        tracker.read('h_pre', hidden)

    h = [max(0.0, v) for v in h_pre]

    if tracker:
        tracker.write('h', hidden)
        tracker.read('h', hidden)
        tracker.read('W2', hidden)
        tracker.read('b2', 1)

    out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]

    if tracker:
        tracker.write('out', 1)

    return out, h_pre, h


def forward_batch(xs, W1, b1, W2, b2):
    """Forward pass for multiple samples. Returns list of outputs."""
    return [forward(x, W1, b1, W2, b2)[0] for x in xs]

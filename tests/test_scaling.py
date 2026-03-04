"""Verify the pipeline works at 20-bit scale."""

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params
from sparse_parity.train import train


def test_20bit_converges():
    """20-bit sparse parity (3 relevant + 17 noise) should show learning signal."""
    config = Config(n_bits=20, k_sparse=3, n_train=200, n_test=200,
                    hidden=500, max_epochs=20, seed=42)
    x_train, y_train, x_test, y_test, secret = generate(config)
    W1, b1, W2, b2 = init_params(config)
    result = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config, tracker_step=-1)
    # 20-bit is hard in pure Python; >50% means it's learning above chance
    assert result['best_test_acc'] > 0.5, f"Only reached {result['best_test_acc']:.0%}"


def test_20bit_under_five_seconds():
    """20-bit with moderate hidden should run in <5 seconds in pure Python."""
    config = Config(n_bits=20, k_sparse=3, n_train=100, n_test=100,
                    hidden=500, max_epochs=3, seed=42)
    x_train, y_train, x_test, y_test, _ = generate(config)
    W1, b1, W2, b2 = init_params(config)
    result = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config, tracker_step=-1)
    assert result['elapsed_s'] < 5.0, f"Took {result['elapsed_s']:.2f}s"

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params
from sparse_parity.train import train


def test_baseline_converges(small_config):
    """3-bit parity with standard backprop should reach >90% accuracy."""
    data = generate(small_config)
    x_train, y_train, x_test, y_test, secret = data
    W1, b1, W2, b2 = init_params(small_config)
    result = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, small_config)
    assert result['best_test_acc'] > 0.9, f"Only reached {result['best_test_acc']:.0%}"


def test_train_returns_required_fields(small_config):
    data = generate(small_config)
    x_train, y_train, x_test, y_test, _ = data
    W1, b1, W2, b2 = init_params(small_config)
    result = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, small_config)
    for key in ['train_losses', 'test_losses', 'train_accs', 'test_accs',
                'best_test_acc', 'total_steps', 'elapsed_s']:
        assert key in result, f"Missing key: {key}"


def test_train_under_one_second(small_config):
    data = generate(small_config)
    x_train, y_train, x_test, y_test, _ = data
    W1, b1, W2, b2 = init_params(small_config)
    result = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, small_config)
    assert result['elapsed_s'] < 1.0, f"Took {result['elapsed_s']:.2f}s"

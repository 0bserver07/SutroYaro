from sparse_parity.config import Config
from sparse_parity.data import generate


def test_generate_returns_correct_shapes(small_config):
    x_train, y_train, x_test, y_test, secret = generate(small_config)
    assert len(x_train) == small_config.n_train
    assert len(y_train) == small_config.n_train
    assert len(x_test) == small_config.n_test
    assert len(y_test) == small_config.n_test
    assert len(secret) == small_config.k_sparse
    assert len(x_train[0]) == small_config.n_bits


def test_labels_match_parity(small_config):
    x_train, y_train, _, _, secret = generate(small_config)
    for x, y in zip(x_train, y_train):
        expected = 1.0
        for idx in secret:
            expected *= 1.0 if x[idx] > 0 else -1.0
        assert y == expected, f"Parity mismatch: x={x}, secret={secret}, got {y}, expected {expected}"


def test_inputs_are_plus_minus_one(small_config):
    x_train, _, x_test, _, _ = generate(small_config)
    for xs in [x_train, x_test]:
        for x in xs:
            for val in x:
                assert val in (-1.0, 1.0)


def test_labels_are_plus_minus_one(small_config):
    _, y_train, _, y_test, _ = generate(small_config)
    for ys in [y_train, y_test]:
        for y in ys:
            assert y in (-1.0, 1.0)


def test_reproducible_with_same_seed(small_config):
    result1 = generate(small_config)
    result2 = generate(small_config)
    assert result1[0] == result2[0]  # x_train identical
    assert result1[4] == result2[4]  # secret identical


def test_secret_indices_in_range(small_config):
    _, _, _, _, secret = generate(small_config)
    for idx in secret:
        assert 0 <= idx < small_config.n_bits

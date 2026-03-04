from sparse_parity.model import init_params, forward


def test_init_params_shapes(small_config):
    W1, b1, W2, b2 = init_params(small_config)
    assert len(W1) == small_config.hidden
    assert len(W1[0]) == small_config.n_bits
    assert len(b1) == small_config.hidden
    assert len(W2) == 1
    assert len(W2[0]) == small_config.hidden
    assert len(b2) == 1


def test_forward_returns_scalar(small_config):
    W1, b1, W2, b2 = init_params(small_config)
    x = [1.0] * small_config.n_bits
    out, h_pre, h = forward(x, W1, b1, W2, b2)
    assert isinstance(out, float)
    assert len(h_pre) == small_config.hidden
    assert len(h) == small_config.hidden


def test_relu_nonnegative(small_config):
    W1, b1, W2, b2 = init_params(small_config)
    x = [1.0, -1.0, 1.0]
    _, _, h = forward(x, W1, b1, W2, b2)
    for val in h:
        assert val >= 0.0


def test_init_reproducible(small_config):
    params1 = init_params(small_config)
    params2 = init_params(small_config)
    assert params1[0] == params2[0]  # W1 identical

"""Phase 4a: Fused layer-wise updates — update each layer immediately after computing its gradients."""

import time

from .model import forward, forward_batch
from .metrics import hinge_loss, accuracy
from .config import Config


def backward_and_update_fused(x, y, out, h_pre, h, W1, b1, W2, b2, config, tracker=None):
    """Fused: grad_layer2 -> update_layer2 -> grad_layer1 -> update_layer1."""
    hidden = len(W1)
    n_bits = len(x)

    if tracker:
        tracker.read('out', 1)
        tracker.read('y', 1)

    margin = out * y
    if margin >= 1.0:
        return

    dout = -y

    if tracker:
        tracker.write('dout', 1)

    # -- Layer 2 backward --
    if tracker:
        tracker.read('dout', 1)
        tracker.read('h', hidden)

    dW2_0 = [dout * h[j] for j in range(hidden)]
    db2_0 = dout

    if tracker:
        tracker.write('dW2', hidden)
        tracker.write('db2', 1)

    # Compute dh BEFORE updating W2
    if tracker:
        tracker.read('W2', hidden)
        tracker.read('dout', 1)

    dh = [W2[0][j] * dout for j in range(hidden)]

    if tracker:
        tracker.write('dh', hidden)

    # -- FUSED: Update W2, b2 immediately --
    if tracker:
        tracker.read('dW2', hidden)
        tracker.read('W2', hidden)

    for j in range(hidden):
        W2[0][j] -= config.lr * (dW2_0[j] + config.wd * W2[0][j])

    if tracker:
        tracker.write('W2', hidden)
        tracker.read('db2', 1)
        tracker.read('b2', 1)

    b2[0] -= config.lr * (db2_0 + config.wd * b2[0])

    if tracker:
        tracker.write('b2', 1)

    # -- ReLU backward --
    if tracker:
        tracker.read('dh', hidden)
        tracker.read('h_pre', hidden)

    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]

    if tracker:
        tracker.write('dh_pre', hidden)

    # -- FUSED: Layer 1 backward + update --
    if tracker:
        tracker.read('dh_pre', hidden)
        tracker.read('x', n_bits)
        tracker.read('W1', hidden * n_bits)

    for j in range(hidden):
        for i in range(n_bits):
            grad = dh_pre[j] * x[i]
            W1[j][i] -= config.lr * (grad + config.wd * W1[j][i])

    if tracker:
        tracker.write('W1', hidden * n_bits)
        tracker.read('dh_pre', hidden)
        tracker.read('b1', hidden)

    for j in range(hidden):
        b1[j] -= config.lr * (dh_pre[j] + config.wd * b1[j])

    if tracker:
        tracker.write('b1', hidden)


def train_fused(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config, tracker_step=0):
    """Train with fused layer-wise updates."""
    from .tracker import MemTracker

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    step = 0
    best_test_acc = 0.0
    tracker_result = None

    start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        for i in range(len(x_train)):
            tracker = MemTracker() if step == tracker_step else None

            if tracker:
                tracker.write('W1', config.hidden * config.n_bits)
                tracker.write('b1', config.hidden)
                tracker.write('W2', config.hidden)
                tracker.write('b2', 1)
                tracker.write('x', config.n_bits)
                tracker.write('y', 1)

            out, h_pre, h = forward(x_train[i], W1, b1, W2, b2, tracker=tracker)
            backward_and_update_fused(x_train[i], y_train[i], out, h_pre, h,
                                      W1, b1, W2, b2, config, tracker=tracker)

            if tracker:
                tracker_result = tracker.to_json()

            step += 1

        tr_outs = forward_batch(x_train, W1, b1, W2, b2)
        te_outs = forward_batch(x_test, W1, b1, W2, b2)
        train_losses.append(hinge_loss(tr_outs, y_train))
        test_losses.append(hinge_loss(te_outs, y_test))
        train_accs.append(accuracy(tr_outs, y_train))
        test_accs.append(accuracy(te_outs, y_test))

        if test_accs[-1] > best_test_acc:
            best_test_acc = test_accs[-1]

        if best_test_acc >= 1.0:
            break

    elapsed = time.time() - start

    return {
        'train_losses': train_losses, 'test_losses': test_losses,
        'train_accs': train_accs, 'test_accs': test_accs,
        'best_test_acc': best_test_acc, 'total_steps': step,
        'elapsed_s': elapsed, 'tracker': tracker_result,
        'method': 'fused_layerwise',
    }

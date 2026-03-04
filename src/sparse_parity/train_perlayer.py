"""Phase 4b: Per-layer forward-backward — update each layer before proceeding to next.

WARNING: This changes the math. Layer 2's forward uses already-updated W1/b1.
This means gradients are computed with respect to different parameters than standard backprop.
The goal is to minimize ARD by keeping parameters in cache between use and update.
"""

import time

from .metrics import hinge_loss, accuracy
from .config import Config


def train_step_perlayer(x, y, W1, b1, W2, b2, config, tracker=None):
    """
    Per-layer forward-backward for one sample.

    Layer 1: forward -> backward -> update W1,b1
    Layer 2: forward (with updated W1,b1) -> backward -> update W2,b2
    """
    hidden = config.hidden
    n_bits = config.n_bits

    # === Layer 1 forward ===
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

    # === Layer 2 forward ===
    if tracker:
        tracker.read('h', hidden)
        tracker.read('W2', hidden)
        tracker.read('b2', 1)

    out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]

    if tracker:
        tracker.write('out', 1)

    # === Check margin ===
    margin = out * y
    if margin >= 1.0:
        return out

    dout = -y

    # === Layer 2 backward + update ===
    if tracker:
        tracker.read('h', hidden)

    dW2_0 = [dout * h[j] for j in range(hidden)]
    db2_0 = dout

    if tracker:
        tracker.read('W2', hidden)

    dh = [W2[0][j] * dout for j in range(hidden)]

    # Update W2, b2 immediately
    if tracker:
        tracker.read('W2', hidden)

    for j in range(hidden):
        W2[0][j] -= config.lr * (dW2_0[j] + config.wd * W2[0][j])

    if tracker:
        tracker.write('W2', hidden)
        tracker.read('b2', 1)

    b2[0] -= config.lr * (db2_0 + config.wd * b2[0])

    if tracker:
        tracker.write('b2', 1)

    # === Layer 1 backward + update ===
    if tracker:
        tracker.read('h_pre', hidden)

    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]

    if tracker:
        tracker.read('x', n_bits)
        tracker.read('W1', hidden * n_bits)

    for j in range(hidden):
        for i in range(n_bits):
            grad = dh_pre[j] * x[i]
            W1[j][i] -= config.lr * (grad + config.wd * W1[j][i])

    if tracker:
        tracker.write('W1', hidden * n_bits)
        tracker.read('b1', hidden)

    for j in range(hidden):
        b1[j] -= config.lr * (dh_pre[j] + config.wd * b1[j])

    if tracker:
        tracker.write('b1', hidden)

    return out


def forward_batch_perlayer(xs, W1, b1, W2, b2, config):
    """Forward-only batch (no updates) for evaluation."""
    outs = []
    for x in xs:
        hidden = config.hidden
        n_bits = config.n_bits
        h_pre = [sum(W1[j][i] * x[i] for i in range(n_bits)) + b1[j] for j in range(hidden)]
        h = [max(0.0, v) for v in h_pre]
        out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]
        outs.append(out)
    return outs


def train_perlayer(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config, tracker_step=0):
    """Train with per-layer forward-backward."""
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

            train_step_perlayer(x_train[i], y_train[i], W1, b1, W2, b2, config, tracker=tracker)

            if tracker:
                tracker_result = tracker.to_json()

            step += 1

        tr_outs = forward_batch_perlayer(x_train, W1, b1, W2, b2, config)
        te_outs = forward_batch_perlayer(x_test, W1, b1, W2, b2, config)
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
        'method': 'per_layer_fwdbwd',
    }

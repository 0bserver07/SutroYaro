#!/usr/bin/env python3
"""
Experiment: NoProp for Sparse Parity

NoProp (Li, Teh, Pascanu 2025, arxiv 2503.24322) trains each layer independently
as a denoiser, inspired by diffusion models. No gradients flow between layers.

Adaptation to sparse parity:
  - Target y in {-1, +1} is noised by flipping with probability t
  - T layers, each assigned a noise level t_l (high -> low)
  - Each layer receives [x | noisy_y] (n+1 inputs) and learns to predict clean y
  - Layers train independently with local MSE loss — no inter-layer backprop
  - Inference: chain layers sequentially, each refining the prediction
  - No learned inverse mappings needed (simpler than TargetProp)

Research questions:
  1. Does the denoising objective help identify the k relevant bits vs SGD?
  2. Does eliminating the backward pass across layers reduce per-step DMD?
  3. Does NoProp hit the same k=5 scaling wall as SGD?

Usage:
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_noprop.py
"""

import sys
import time
import json
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparse_parity.tracker import MemTracker

# =============================================================================
# CONFIG
# =============================================================================

EXP_NAME = "exp_noprop"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / EXP_NAME

LR = 0.05          # MSE gradients ~2x larger than hinge, so half of SGD's 0.1
WD = 0.01
BATCH_SIZE = 32
HIDDEN = 200
N_TRAIN = 2000
N_TEST = 200
SEEDS = [42, 43, 44, 45, 46]

N_LAYERS = 5
NOISE_SCHEDULE = [0.9, 0.7, 0.5, 0.3, 0.1]  # noise prob per layer, high -> low

FF_LR = 0.01          # Forward-Forward learning rate (contrastive goodness loss)
FF_THRESHOLD = 2.0    # FF goodness threshold


# =============================================================================
# DATA
# =============================================================================

def generate_data(n_bits, k_sparse, secret, n_samples, rng):
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    return x, y


# =============================================================================
# NOPROP PARAMS
# =============================================================================

def init_noprop_params(n_bits, hidden, n_layers, seed):
    """Initialize T independent sets of (W1, b1, W2, b2).
    Each layer receives n_bits+1 inputs: [x | noisy_y].
    """
    params = []
    for l in range(n_layers):
        rng = np.random.RandomState(seed + 1 + l * 100)
        n_in = n_bits + 1
        std1 = math.sqrt(2.0 / n_in)
        std2 = math.sqrt(2.0 / hidden)
        W1 = rng.randn(hidden, n_in) * std1
        b1 = np.zeros(hidden)
        W2 = rng.randn(1, hidden) * std2
        b2 = np.zeros(1)
        params.append((W1, b1, W2, b2))
    return params


# =============================================================================
# NOISE
# =============================================================================

def apply_noise(y, noise_prob, rng):
    """Flip each label in {-1, +1} independently with probability noise_prob."""
    flips = rng.rand(len(y)) < noise_prob
    return np.where(flips, -y, y)


# =============================================================================
# LAYER FORWARD + UPDATE
# =============================================================================

def layer_forward(x_aug, W1, b1, W2, b2):
    """Forward pass for one NoProp layer.
    x_aug: (bs, n_bits+1) — x concatenated with noisy_y
    Returns pred (bs,), h_pre (bs, hidden), h (bs, hidden)
    """
    h_pre = x_aug @ W1.T + b1       # (bs, hidden)
    h = np.maximum(h_pre, 0.0)      # ReLU
    pred = (h @ W2.T + b2).ravel()  # (bs,)
    return pred, h_pre, h


def layer_update(x_aug, y_clean, pred, h_pre, h, W1, b1, W2, b2):
    """MSE gradient step for one NoProp layer (local only — no inter-layer grads).
    x_aug: (bs, n_bits+1)
    y_clean: (bs,) — true labels (not noisy)
    """
    bs = len(y_clean)
    err = pred - y_clean            # (bs,) MSE error

    # Layer 2 gradients
    gW2 = (err[:, None] * h).sum(axis=0, keepdims=True) / bs   # (1, hidden)
    gb2 = err.sum() / bs                                         # scalar

    # Backprop through layer 2 into hidden
    dh = err[:, None] * W2          # (bs, hidden)
    dh_pre = dh * (h_pre > 0)       # ReLU mask (bs, hidden)

    # Layer 1 gradients
    gW1 = (dh_pre.T @ x_aug) / bs   # (hidden, n_bits+1)
    gb1 = dh_pre.sum(axis=0) / bs   # (hidden,)

    # Weight decay + update
    W2 -= LR * (gW2 + WD * W2)
    b2 -= LR * gb2
    W1 -= LR * (gW1 + WD * W1)
    b1 -= LR * gb1

    return W1, b1, W2, b2


# =============================================================================
# INFERENCE
# =============================================================================

def noprop_infer(x, params):
    """Chain layers at inference: start from y=0, refine through each layer.
    x: (bs, n_bits)
    Returns pred (bs,) — final layer's prediction
    """
    bs = x.shape[0]
    y_current = np.zeros(bs)  # neutral start

    for (W1, b1, W2, b2) in params:
        x_aug = np.concatenate([x, y_current[:, None]], axis=1)
        pred, _, _ = layer_forward(x_aug, W1, b1, W2, b2)
        y_current = pred

    return y_current


# =============================================================================
# CURRICULUM HELPERS
# =============================================================================

def expand_noprop_params(params, old_n, new_n, rng_seed):
    """Expand all T layers' W1 from n_in=old_n+1 to new_n+1.
    New input weights are initialized small (matching exp_grokfast_curriculum pattern).
    """
    new_params = []
    for l, (W1, b1, W2, b2) in enumerate(params):
        rng = np.random.RandomState(rng_seed + l * 100)
        old_n_in = old_n + 1
        new_n_in = new_n + 1
        new_std = math.sqrt(2.0 / new_n_in) * 0.1
        W1_new = np.zeros((W1.shape[0], new_n_in))
        W1_new[:, :old_n_in] = W1
        W1_new[:, old_n_in:] = rng.randn(W1.shape[0], new_n_in - old_n_in) * new_std
        new_params.append((W1_new, b1, W2, b2))
    return new_params


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_noprop(n_bits, k_sparse, secret, seed, max_epochs=500, target_acc=0.95):
    """Train NoProp across T independent layers. Returns result dict."""
    rng = np.random.RandomState(seed)
    x_tr, y_tr = generate_data(n_bits, k_sparse, secret, N_TRAIN, rng)
    x_te, y_te = generate_data(n_bits, k_sparse, secret, N_TEST, rng)

    params = init_noprop_params(n_bits, HIDDEN, N_LAYERS, seed)

    # Per-layer noise RNGs (seeded independently per layer)
    noise_rngs = [np.random.RandomState(seed + 200 + l * 100) for l in range(N_LAYERS)]

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(N_TRAIN)
        rng.shuffle(idx)
        x_sh, y_sh = x_tr[idx], y_tr[idx]

        # Train each layer independently
        for l, (noise_prob, noise_rng) in enumerate(zip(NOISE_SCHEDULE, noise_rngs)):
            W1, b1, W2, b2 = params[l]

            for b_start in range(0, N_TRAIN, BATCH_SIZE):
                b_end = min(b_start + BATCH_SIZE, N_TRAIN)
                xb = x_sh[b_start:b_end]
                yb = y_sh[b_start:b_end]

                # Apply fresh noise each mini-batch for stochasticity
                noisy_yb = apply_noise(yb, noise_prob, noise_rng)
                x_aug = np.concatenate([xb, noisy_yb[:, None]], axis=1)

                pred, h_pre, h = layer_forward(x_aug, W1, b1, W2, b2)
                W1, b1, W2, b2 = layer_update(x_aug, yb, pred, h_pre, h, W1, b1, W2, b2)

            params[l] = (W1, b1, W2, b2)

        # Evaluate via chained inference
        te_pred = noprop_infer(x_te, params)
        te_acc = float(np.mean(np.sign(te_pred) == y_te))

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= target_acc and solve_epoch < 0:
            solve_epoch = epoch
            break

    elapsed = time.time() - start
    return {
        'best_test_acc': best_acc,
        'solve_epoch': solve_epoch,
        'total_epochs': epoch,
        'elapsed_s': round(elapsed, 4),
    }


def train_noprop_curriculum(stages, k_sparse, secret, seed,
                             max_epochs_per_phase=500, target_acc=0.95):
    """NoProp with n-curriculum: train on small n, expand, repeat."""
    total_start = time.time()
    total_epochs = 0

    n0 = stages[0]
    params = init_noprop_params(n0, HIDDEN, N_LAYERS, seed)
    noise_rngs = [np.random.RandomState(seed + 200 + l * 100) for l in range(N_LAYERS)]

    def _run_phase(params, n_bits, rng_seed):
        rng = np.random.RandomState(rng_seed)
        x_tr, y_tr = generate_data(n_bits, k_sparse, secret, N_TRAIN, rng)
        x_te, y_te = generate_data(n_bits, k_sparse, secret, N_TEST, rng)
        best_acc = 0.0
        solve_epoch = -1
        epoch = 0
        for epoch in range(1, max_epochs_per_phase + 1):
            idx = np.arange(N_TRAIN)
            rng.shuffle(idx)
            x_sh, y_sh = x_tr[idx], y_tr[idx]
            for l, noise_prob in enumerate(NOISE_SCHEDULE):
                W1, b1, W2, b2 = params[l]
                for b_start in range(0, N_TRAIN, BATCH_SIZE):
                    b_end = min(b_start + BATCH_SIZE, N_TRAIN)
                    xb = x_sh[b_start:b_end]
                    yb = y_sh[b_start:b_end]
                    noisy_yb = apply_noise(yb, noise_prob, noise_rngs[l])
                    x_aug = np.concatenate([xb, noisy_yb[:, None]], axis=1)
                    pred, h_pre, h = layer_forward(x_aug, W1, b1, W2, b2)
                    W1, b1, W2, b2 = layer_update(x_aug, yb, pred, h_pre, h, W1, b1, W2, b2)
                params[l] = (W1, b1, W2, b2)
            te_pred = noprop_infer(x_te, params)
            te_acc = float(np.mean(np.sign(te_pred) == y_te))
            if te_acc > best_acc:
                best_acc = te_acc
            if te_acc >= target_acc and solve_epoch < 0:
                solve_epoch = epoch
                break
        return params, best_acc, epoch

    params, _, phase_epochs = _run_phase(params, n0, seed)
    total_epochs += phase_epochs

    for i, n_next in enumerate(stages[1:], start=2):
        n_prev = stages[i - 2]
        params = expand_noprop_params(params, n_prev, n_next, rng_seed=seed + i * 100)
        params, best_acc, phase_epochs = _run_phase(params, n_next, seed + i * 1000)
        total_epochs += phase_epochs

    total_elapsed = time.time() - total_start
    return {
        'best_test_acc': best_acc,
        'total_epochs': total_epochs,
        'elapsed_s': round(total_elapsed, 4),
    }


# =============================================================================
# SGD BASELINE + CURRICULUM (hinge loss, matches exp_grokfast_curriculum.py)
# =============================================================================

def train_sgd(n_bits, k_sparse, secret, seed, max_epochs=500, target_acc=0.95):
    """Standard SGD baseline for comparison."""
    rng = np.random.RandomState(seed)
    x_tr, y_tr = generate_data(n_bits, k_sparse, secret, N_TRAIN, rng)
    x_te, y_te = generate_data(n_bits, k_sparse, secret, N_TEST, rng)

    rng_w = np.random.RandomState(seed + 1)
    std1 = math.sqrt(2.0 / n_bits)
    std2 = math.sqrt(2.0 / HIDDEN)
    W1 = rng_w.randn(HIDDEN, n_bits) * std1
    b1 = np.zeros(HIDDEN)
    W2 = rng_w.randn(1, HIDDEN) * std2
    b2 = np.zeros(1)

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1

    SGD_LR = 0.1  # standard SGD LR from prior experiments

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(N_TRAIN)
        rng.shuffle(idx)

        for b_start in range(0, N_TRAIN, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, N_TRAIN)
            xb = x_tr[idx[b_start:b_end]]
            yb = y_tr[idx[b_start:b_end]]
            bs = xb.shape[0]

            h_pre = xb @ W1.T + b1
            h = np.maximum(h_pre, 0)
            out = (h @ W2.T + b2).ravel()

            margin = out * yb
            mask = margin < 1.0
            if not np.any(mask):
                continue

            xm, ym, hm, h_pre_m = xb[mask], yb[mask], h[mask], h_pre[mask]
            dout = -ym
            gW2 = (dout[:, None] * hm).sum(axis=0, keepdims=True) / bs
            gb2 = dout.sum() / bs
            dh = dout[:, None] * W2
            dh_pre = dh * (h_pre_m > 0)
            gW1 = (dh_pre.T @ xm) / bs
            gb1 = dh_pre.sum(axis=0) / bs

            W2 -= SGD_LR * (gW2 + WD * W2)
            b2 -= SGD_LR * (gb2 + WD * b2)
            W1 -= SGD_LR * (gW1 + WD * W1)
            b1 -= SGD_LR * (gb1 + WD * b1)

        te_out = (np.maximum(x_te @ W1.T + b1, 0) @ W2.T + b2).ravel()
        te_acc = float(np.mean(np.sign(te_out) == y_te))

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= target_acc and solve_epoch < 0:
            solve_epoch = epoch
            break

    elapsed = time.time() - start
    return {
        'best_test_acc': best_acc,
        'solve_epoch': solve_epoch,
        'total_epochs': epoch,
        'elapsed_s': round(elapsed, 4),
    }


def train_sgd_curriculum(stages, k_sparse, secret, seed,
                          max_epochs_per_phase=500, target_acc=0.95):
    """SGD with n-curriculum (no GrokFast). Mirrors train_noprop_curriculum."""
    SGD_LR = 0.1
    total_start = time.time()
    total_epochs = 0

    def _init(n_bits, seed):
        rng = np.random.RandomState(seed + 1)
        std1 = math.sqrt(2.0 / n_bits)
        std2 = math.sqrt(2.0 / HIDDEN)
        W1 = rng.randn(HIDDEN, n_bits) * std1
        b1 = np.zeros(HIDDEN)
        W2 = rng.randn(1, HIDDEN) * std2
        b2 = np.zeros(1)
        return W1, b1, W2, b2

    def _expand(W1, old_n, new_n, rng_seed):
        rng = np.random.RandomState(rng_seed)
        new_std = math.sqrt(2.0 / new_n) * 0.1
        W1_new = np.zeros((HIDDEN, new_n))
        W1_new[:, :old_n] = W1
        W1_new[:, old_n:] = rng.randn(HIDDEN, new_n - old_n) * new_std
        return W1_new

    def _run_phase(W1, b1, W2, b2, n_bits, rng_seed):
        rng = np.random.RandomState(rng_seed)
        x_tr, y_tr = generate_data(n_bits, k_sparse, secret, N_TRAIN, rng)
        x_te, y_te = generate_data(n_bits, k_sparse, secret, N_TEST, rng)
        best_acc = 0.0
        epoch = 0
        for epoch in range(1, max_epochs_per_phase + 1):
            idx = np.arange(N_TRAIN)
            rng.shuffle(idx)
            for b_start in range(0, N_TRAIN, BATCH_SIZE):
                b_end = min(b_start + BATCH_SIZE, N_TRAIN)
                xb = x_tr[idx[b_start:b_end]]
                yb = y_tr[idx[b_start:b_end]]
                bs = xb.shape[0]
                h_pre = xb @ W1.T + b1
                h = np.maximum(h_pre, 0)
                out = (h @ W2.T + b2).ravel()
                margin = out * yb
                mask = margin < 1.0
                if not np.any(mask):
                    continue
                xm, ym, hm, h_pre_m = xb[mask], yb[mask], h[mask], h_pre[mask]
                dout = -ym
                gW2 = (dout[:, None] * hm).sum(axis=0, keepdims=True) / bs
                gb2 = dout.sum() / bs
                dh = dout[:, None] * W2
                dh_pre = dh * (h_pre_m > 0)
                gW1 = (dh_pre.T @ xm) / bs
                gb1 = dh_pre.sum(axis=0) / bs
                W2 -= SGD_LR * (gW2 + WD * W2)
                b2 -= SGD_LR * (gb2 + WD * b2)
                W1 -= SGD_LR * (gW1 + WD * W1)
                b1 -= SGD_LR * (gb1 + WD * b1)
            te_out = (np.maximum(x_te @ W1.T + b1, 0) @ W2.T + b2).ravel()
            te_acc = float(np.mean(np.sign(te_out) == y_te))
            if te_acc > best_acc:
                best_acc = te_acc
            if te_acc >= target_acc:
                break
        return W1, b1, W2, b2, best_acc, epoch

    n0 = stages[0]
    W1, b1, W2, b2 = _init(n0, seed)
    W1, b1, W2, b2, _, phase_epochs = _run_phase(W1, b1, W2, b2, n0, seed)
    total_epochs += phase_epochs

    for i, n_next in enumerate(stages[1:], start=2):
        n_prev = stages[i - 2]
        W1 = _expand(W1, n_prev, n_next, rng_seed=seed + i * 100)
        W1, b1, W2, b2, best_acc, phase_epochs = _run_phase(W1, b1, W2, b2, n_next, seed + i * 1000)
        total_epochs += phase_epochs

    return {
        'best_test_acc': best_acc,
        'total_epochs': total_epochs,
        'elapsed_s': round(time.time() - total_start, 4),
    }


# =============================================================================
# FORWARD-FORWARD BASELINE (Hinton 2022, contrastive local learning)
#
# FF embeds the label as the first input dimension, then trains each layer
# to maximize goodness (sum of squared ReLU activations) on positive data
# (correct label) and minimize it on negative data (wrong label).
# Key distinction from NoProp: corrupts input x framing (label embedding)
# rather than label y directly, and uses a contrastive objective.
# =============================================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def init_ff_params(n_bits, hidden, seed):
    """Initialize 2-layer FF network. Input = [label | x], n_in = n_bits+1.
    Returns W1 (hidden, n_bits+1), b1, W2 (hidden, hidden), b2.
    """
    rng = np.random.RandomState(seed + 300)
    n_in = n_bits + 1
    W1 = rng.randn(hidden, n_in) * math.sqrt(2.0 / n_in)
    b1 = np.zeros(hidden)
    W2 = rng.randn(hidden, hidden) * math.sqrt(2.0 / hidden)
    b2 = np.zeros(hidden)
    return W1, b1, W2, b2


def ff_forward(x_lab, W1, b1, W2, b2):
    """Forward pass through both FF layers.
    x_lab: (bs, n_bits+1). Returns h1_pre, h1, h1_norm, h2_pre, h2.
    """
    h1_pre = x_lab @ W1.T + b1
    h1 = np.maximum(h1_pre, 0.0)
    h1_norm = h1 / (np.linalg.norm(h1, axis=1, keepdims=True) + 1e-8)
    h2_pre = h1_norm @ W2.T + b2
    h2 = np.maximum(h2_pre, 0.0)
    return h1_pre, h1, h1_norm, h2_pre, h2


def ff_layer_update(x_in, W, b, h_pre, h, is_positive, lr, wd, threshold):
    """Vectorized FF update: soft contrastive loss on goodness.
    Loss = -log(sigmoid(sign * (goodness - threshold))).
    x_in: (bs, n_in), h/h_pre: (bs, hidden).
    """
    goodness = (h ** 2).sum(axis=1)                     # (bs,)
    p = _sigmoid(goodness - threshold)                   # (bs,)
    dloss_dg = (p - 1.0) if is_positive else p           # (bs,)

    relu_mask = (h_pre > 0).astype(float)                # (bs, hidden)
    dg_dh = 2.0 * h * relu_mask                          # (bs, hidden)
    factor = dloss_dg[:, None] * dg_dh                   # (bs, hidden)

    gW = (factor.T @ x_in) / len(h)                      # (hidden, n_in)
    gb = factor.mean(axis=0)                              # (hidden,)

    W -= lr * (gW + wd * W)
    b -= lr * gb
    return W, b


def ff_train_step(xb, yb, W1, b1, W2, b2, lr, wd, threshold):
    """One FF minibatch: positive pass (correct label) then negative pass."""
    x_pos = np.concatenate([yb[:, None], xb], axis=1)
    h1_pre_p, h1_p, h1_norm_p, h2_pre_p, h2_p = ff_forward(x_pos, W1, b1, W2, b2)
    W1, b1 = ff_layer_update(x_pos, W1, b1, h1_pre_p, h1_p, True, lr, wd, threshold)
    W2, b2 = ff_layer_update(h1_norm_p, W2, b2, h2_pre_p, h2_p, True, lr, wd, threshold)

    x_neg = np.concatenate([-yb[:, None], xb], axis=1)
    h1_pre_n, h1_n, h1_norm_n, h2_pre_n, h2_n = ff_forward(x_neg, W1, b1, W2, b2)
    W1, b1 = ff_layer_update(x_neg, W1, b1, h1_pre_n, h1_n, False, lr, wd, threshold)
    W2, b2 = ff_layer_update(h1_norm_n, W2, b2, h2_pre_n, h2_n, False, lr, wd, threshold)

    return W1, b1, W2, b2


def ff_predict(x, W1, b1, W2, b2):
    """Predict by comparing total goodness for label=+1 vs label=-1."""
    bs = x.shape[0]
    x_pos = np.concatenate([np.ones((bs, 1)), x], axis=1)
    _, h1_p, _, _, h2_p = ff_forward(x_pos, W1, b1, W2, b2)
    g_pos = (h1_p ** 2).sum(axis=1) + (h2_p ** 2).sum(axis=1)

    x_neg = np.concatenate([-np.ones((bs, 1)), x], axis=1)
    _, h1_n, _, _, h2_n = ff_forward(x_neg, W1, b1, W2, b2)
    g_neg = (h1_n ** 2).sum(axis=1) + (h2_n ** 2).sum(axis=1)

    return np.where(g_pos >= g_neg, 1.0, -1.0)


def expand_ff_params(W1, b1, W2, b2, old_n, new_n, rng_seed):
    """Expand W1 from (hidden, old_n+1) to (hidden, new_n+1).
    Col 0 is the label channel — preserved. New x-feature cols init small.
    """
    rng = np.random.RandomState(rng_seed)
    new_std = math.sqrt(2.0 / (new_n + 1)) * 0.1
    W1_new = np.zeros((W1.shape[0], new_n + 1))
    W1_new[:, :old_n + 1] = W1
    W1_new[:, old_n + 1:] = rng.randn(W1.shape[0], new_n - old_n) * new_std
    return W1_new, b1, W2, b2


def train_ff(n_bits, k_sparse, secret, seed, max_epochs=500, target_acc=0.95):
    """Train Forward-Forward network. Returns result dict."""
    rng = np.random.RandomState(seed)
    x_tr, y_tr = generate_data(n_bits, k_sparse, secret, N_TRAIN, rng)
    x_te, y_te = generate_data(n_bits, k_sparse, secret, N_TEST, rng)

    W1, b1, W2, b2 = init_ff_params(n_bits, HIDDEN, seed)
    start = time.time()
    best_acc = 0.0
    solve_epoch = -1

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(N_TRAIN)
        rng.shuffle(idx)
        x_sh, y_sh = x_tr[idx], y_tr[idx]

        for b_start in range(0, N_TRAIN, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, N_TRAIN)
            W1, b1, W2, b2 = ff_train_step(
                x_sh[b_start:b_end], y_sh[b_start:b_end],
                W1, b1, W2, b2, FF_LR, WD, FF_THRESHOLD)

        te_pred = ff_predict(x_te, W1, b1, W2, b2)
        te_acc = float(np.mean(te_pred == y_te))

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= target_acc and solve_epoch < 0:
            solve_epoch = epoch
            break

    return {
        'best_test_acc': best_acc,
        'solve_epoch': solve_epoch,
        'total_epochs': epoch,
        'elapsed_s': round(time.time() - start, 4),
    }


def train_ff_curriculum(stages, k_sparse, secret, seed,
                         max_epochs_per_phase=500, target_acc=0.95):
    """Forward-Forward with n-curriculum."""
    total_start = time.time()
    total_epochs = 0

    n0 = stages[0]
    W1, b1, W2, b2 = init_ff_params(n0, HIDDEN, seed)

    def _run_phase(W1, b1, W2, b2, n_bits, rng_seed):
        rng = np.random.RandomState(rng_seed)
        x_tr, y_tr = generate_data(n_bits, k_sparse, secret, N_TRAIN, rng)
        x_te, y_te = generate_data(n_bits, k_sparse, secret, N_TEST, rng)
        best_acc = 0.0
        epoch = 0
        for epoch in range(1, max_epochs_per_phase + 1):
            idx = np.arange(N_TRAIN)
            rng.shuffle(idx)
            x_sh, y_sh = x_tr[idx], y_tr[idx]
            for b_start in range(0, N_TRAIN, BATCH_SIZE):
                b_end = min(b_start + BATCH_SIZE, N_TRAIN)
                W1, b1, W2, b2 = ff_train_step(
                    x_sh[b_start:b_end], y_sh[b_start:b_end],
                    W1, b1, W2, b2, FF_LR, WD, FF_THRESHOLD)
            te_pred = ff_predict(x_te, W1, b1, W2, b2)
            te_acc = float(np.mean(te_pred == y_te))
            if te_acc > best_acc:
                best_acc = te_acc
            if te_acc >= target_acc:
                break
        return W1, b1, W2, b2, best_acc, epoch

    W1, b1, W2, b2, _, phase_epochs = _run_phase(W1, b1, W2, b2, n0, seed)
    total_epochs += phase_epochs

    for i, n_next in enumerate(stages[1:], start=2):
        n_prev = stages[i - 2]
        W1, b1, W2, b2 = expand_ff_params(W1, b1, W2, b2, n_prev, n_next, rng_seed=seed + i * 100)
        W1, b1, W2, b2, best_acc, phase_epochs = _run_phase(W1, b1, W2, b2, n_next, seed + i * 1000)
        total_epochs += phase_epochs

    return {
        'best_test_acc': best_acc,
        'total_epochs': total_epochs,
        'elapsed_s': round(time.time() - total_start, 4),
    }


# =============================================================================
# DMD MEASUREMENT
# =============================================================================

def measure_noprop_dmd(n_bits, hidden):
    """Measure per-step DMD for one NoProp layer forward+update pass."""
    tracker = MemTracker()
    n_in = n_bits + 1

    tracker.write('W1', hidden * n_in)
    tracker.write('b1', hidden)
    tracker.write('W2', hidden)
    tracker.write('b2', 1)
    tracker.write('x_aug', n_in)   # [x | noisy_y] for one sample
    tracker.write('y', 1)

    # Forward
    tracker.read('x_aug', n_in)
    tracker.read('W1', hidden * n_in)
    tracker.read('b1', hidden)
    tracker.write('h_pre', hidden)

    tracker.read('h_pre', hidden)
    tracker.write('h', hidden)

    tracker.read('h', hidden)
    tracker.read('W2', hidden)
    tracker.read('b2', 1)
    tracker.write('pred', 1)

    # Backward (local only — no inter-layer)
    tracker.read('pred', 1)
    tracker.read('y', 1)
    tracker.write('err', 1)

    tracker.read('err', 1)
    tracker.read('h', hidden)
    tracker.write('gW2', hidden)
    tracker.write('gb2', 1)

    tracker.read('err', 1)
    tracker.read('W2', hidden)
    tracker.write('dh', hidden)

    tracker.read('dh', hidden)
    tracker.read('h_pre', hidden)
    tracker.write('dh_pre', hidden)

    tracker.read('dh_pre', hidden)
    tracker.read('x_aug', n_in)
    tracker.write('gW1', hidden * n_in)
    tracker.write('gb1', hidden)

    # Update
    tracker.read('W2', hidden)
    tracker.read('gW2', hidden)
    tracker.write('W2', hidden)
    tracker.read('W1', hidden * n_in)
    tracker.read('gW1', hidden * n_in)
    tracker.write('W1', hidden * n_in)

    return tracker.to_json()


def measure_sgd_dmd(n_bits, hidden):
    """Measure per-step DMD for one SGD forward+backward pass."""
    tracker = MemTracker()

    tracker.write('W1', hidden * n_bits)
    tracker.write('b1', hidden)
    tracker.write('W2', hidden)
    tracker.write('b2', 1)
    tracker.write('x', n_bits)
    tracker.write('y', 1)

    # Forward
    tracker.read('x', n_bits)
    tracker.read('W1', hidden * n_bits)
    tracker.read('b1', hidden)
    tracker.write('h_pre', hidden)

    tracker.read('h_pre', hidden)
    tracker.write('h', hidden)

    tracker.read('h', hidden)
    tracker.read('W2', hidden)
    tracker.read('b2', 1)
    tracker.write('out', 1)

    # Backward
    tracker.read('out', 1)
    tracker.read('y', 1)
    tracker.write('dout', 1)

    tracker.read('dout', 1)
    tracker.read('h', hidden)
    tracker.write('gW2', hidden)

    tracker.read('dout', 1)
    tracker.read('W2', hidden)
    tracker.write('dh', hidden)

    tracker.read('dh', hidden)
    tracker.read('h_pre', hidden)
    tracker.write('dh_pre', hidden)

    tracker.read('dh_pre', hidden)
    tracker.read('x', n_bits)
    tracker.write('gW1', hidden * n_bits)
    tracker.write('gb1', hidden)

    # Update
    tracker.read('W2', hidden)
    tracker.read('gW2', hidden)
    tracker.write('W2', hidden)
    tracker.read('W1', hidden * n_bits)
    tracker.read('gW1', hidden * n_bits)
    tracker.write('W1', hidden * n_bits)

    return tracker.to_json()


def measure_ff_dmd(n_bits, hidden):
    """Measure per-step ARD for one FF positive+negative pass (2-layer, 2 passes).
    FF does 2 forward passes (pos + neg), each 2 layers, with local updates only.
    """
    tracker = MemTracker()
    n_in = n_bits + 1  # [label | x]

    tracker.write('W1', hidden * n_in)
    tracker.write('b1', hidden)
    tracker.write('W2', hidden * hidden)
    tracker.write('b2', hidden)
    tracker.write('x', n_bits)
    tracker.write('y', 1)

    for _pass in range(2):  # positive pass + negative pass
        # Forward layer 1
        tracker.read('x', n_bits)
        tracker.read('y', 1)                   # label embedding
        tracker.read('W1', hidden * n_in)
        tracker.read('b1', hidden)
        tracker.write('h1', hidden)

        # Normalize h1, then forward layer 2
        tracker.read('h1', hidden)
        tracker.read('W2', hidden * hidden)
        tracker.read('b2', hidden)
        tracker.write('h2', hidden)

        # Local update layer 1 (goodness of h1)
        tracker.read('h1', hidden)
        tracker.write('gW1', hidden * n_in)
        tracker.write('gb1', hidden)
        tracker.read('W1', hidden * n_in)
        tracker.read('gW1', hidden * n_in)
        tracker.write('W1', hidden * n_in)

        # Local update layer 2 (goodness of h2)
        tracker.read('h2', hidden)
        tracker.write('gW2', hidden * hidden)
        tracker.write('gb2', hidden)
        tracker.read('W2', hidden * hidden)
        tracker.read('gW2', hidden * hidden)
        tracker.write('W2', hidden * hidden)

    return tracker.to_json()


# =============================================================================
# MULTI-SEED RUNNER
# =============================================================================

def run_multi_seed(runner, seeds, label, **kwargs):
    accs, epochs, times = [], [], []
    for seed in seeds:
        r = runner(seed=seed, **kwargs)
        accs.append(r['best_test_acc'])
        ep = r.get('solve_epoch', r['total_epochs'])
        if ep < 0:
            ep = r['total_epochs']
        epochs.append(ep)
        times.append(r['elapsed_s'])

    solve_rate = sum(1 for a in accs if a >= 0.95) / len(seeds)
    avg_ep = sum(epochs) / len(epochs)
    avg_t = sum(times) / len(times)
    print(f"  {label:<45} solve={solve_rate:.0%}  epochs={avg_ep:>6.0f}  time={avg_t:.3f}s")

    return {
        'label': label,
        'solve_rate': solve_rate,
        'avg_epochs': avg_ep,
        'avg_time': avg_t,
        'accs': accs,
        'epochs': epochs,
        'times': times,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  EXPERIMENT: NoProp vs Forward-Forward for Sparse Parity")
    print(f"  NoProp: T={N_LAYERS} layers, noise={NOISE_SCHEDULE}")
    print(f"  FF: 2-layer contrastive, lr={FF_LR}, threshold={FF_THRESHOLD}")
    print(f"  7 methods x 3 regimes x 5 seeds = 105 runs")
    print("=" * 78)

    all_results = {}
    rng = np.random.RandomState(42)

    # --- Regime 1: n=20, k=5 (high interaction order) ---
    print(f"\n{'─'*78}")
    print(f"  REGIME: n=20, k=5 (high interaction order)")
    print(f"{'─'*78}")

    secret_k5_20 = sorted(rng.choice(10, 5, replace=False).tolist())
    print(f"  Secret: {secret_k5_20}\n")

    r1 = {}
    r1['sgd'] = run_multi_seed(
        train_sgd, SEEDS, "SGD (hinge)",
        n_bits=20, k_sparse=5, secret=secret_k5_20, max_epochs=500)
    r1['noprop'] = run_multi_seed(
        train_noprop, SEEDS, f"NoProp (T={N_LAYERS}, noise={NOISE_SCHEDULE})",
        n_bits=20, k_sparse=5, secret=secret_k5_20, max_epochs=500)
    r1['sgd_curriculum'] = run_multi_seed(
        train_sgd_curriculum, SEEDS, "SGD + Curriculum [10->20]",
        stages=[10, 20], k_sparse=5, secret=secret_k5_20)
    r1['noprop_curriculum'] = run_multi_seed(
        train_noprop_curriculum, SEEDS, f"NoProp + Curriculum [10->20]",
        stages=[10, 20], k_sparse=5, secret=secret_k5_20)
    r1['ff'] = run_multi_seed(
        train_ff, SEEDS, f"FF (lr={FF_LR}, threshold={FF_THRESHOLD})",
        n_bits=20, k_sparse=5, secret=secret_k5_20, max_epochs=500)
    r1['ff_curriculum'] = run_multi_seed(
        train_ff_curriculum, SEEDS, "FF + Curriculum [10->20]",
        stages=[10, 20], k_sparse=5, secret=secret_k5_20)
    all_results['n20_k5'] = r1

    # --- Regime 2: n=50, k=3 (high input dimension) ---
    print(f"\n{'─'*78}")
    print(f"  REGIME: n=50, k=3 (high input dimension)")
    print(f"{'─'*78}")

    secret_k3_50 = sorted(rng.choice(10, 3, replace=False).tolist())
    print(f"  Secret: {secret_k3_50}\n")

    r2 = {}
    r2['sgd'] = run_multi_seed(
        train_sgd, SEEDS, "SGD (hinge)",
        n_bits=50, k_sparse=3, secret=secret_k3_50, max_epochs=500)
    r2['noprop'] = run_multi_seed(
        train_noprop, SEEDS, f"NoProp (T={N_LAYERS}, noise={NOISE_SCHEDULE})",
        n_bits=50, k_sparse=3, secret=secret_k3_50, max_epochs=500)
    r2['sgd_curriculum'] = run_multi_seed(
        train_sgd_curriculum, SEEDS, "SGD + Curriculum [10->30->50]",
        stages=[10, 30, 50], k_sparse=3, secret=secret_k3_50)
    r2['noprop_curriculum'] = run_multi_seed(
        train_noprop_curriculum, SEEDS, f"NoProp + Curriculum [10->30->50]",
        stages=[10, 30, 50], k_sparse=3, secret=secret_k3_50)
    r2['ff'] = run_multi_seed(
        train_ff, SEEDS, f"FF (lr={FF_LR}, threshold={FF_THRESHOLD})",
        n_bits=50, k_sparse=3, secret=secret_k3_50, max_epochs=500)
    r2['ff_curriculum'] = run_multi_seed(
        train_ff_curriculum, SEEDS, "FF + Curriculum [10->30->50]",
        stages=[10, 30, 50], k_sparse=3, secret=secret_k3_50)
    all_results['n50_k3'] = r2

    # --- Regime 3: n=50, k=5 (both hard) ---
    print(f"\n{'─'*78}")
    print(f"  REGIME: n=50, k=5 (both hard — the real test)")
    print(f"{'─'*78}")

    secret_k5_50 = sorted(rng.choice(10, 5, replace=False).tolist())
    print(f"  Secret: {secret_k5_50}\n")

    r3 = {}
    r3['sgd'] = run_multi_seed(
        train_sgd, SEEDS, "SGD (hinge)",
        n_bits=50, k_sparse=5, secret=secret_k5_50, max_epochs=1000)
    r3['noprop'] = run_multi_seed(
        train_noprop, SEEDS, f"NoProp (T={N_LAYERS}, noise={NOISE_SCHEDULE})",
        n_bits=50, k_sparse=5, secret=secret_k5_50, max_epochs=1000)
    r3['sgd_curriculum'] = run_multi_seed(
        train_sgd_curriculum, SEEDS, "SGD + Curriculum [10->30->50]",
        stages=[10, 30, 50], k_sparse=5, secret=secret_k5_50, max_epochs_per_phase=500)
    r3['noprop_curriculum'] = run_multi_seed(
        train_noprop_curriculum, SEEDS, f"NoProp + Curriculum [10->30->50]",
        stages=[10, 30, 50], k_sparse=5, secret=secret_k5_50, max_epochs_per_phase=500)
    r3['ff'] = run_multi_seed(
        train_ff, SEEDS, f"FF (lr={FF_LR}, threshold={FF_THRESHOLD})",
        n_bits=50, k_sparse=5, secret=secret_k5_50, max_epochs=1000)
    r3['ff_curriculum'] = run_multi_seed(
        train_ff_curriculum, SEEDS, "FF + Curriculum [10->30->50]",
        stages=[10, 30, 50], k_sparse=5, secret=secret_k5_50, max_epochs_per_phase=500)
    all_results['n50_k5'] = r3

    # --- DMD comparison ---
    print(f"\n{'─'*78}")
    print(f"  DMD COMPARISON (per-step, n=20, hidden={HIDDEN})")
    print(f"{'─'*78}")

    sgd_dmd = measure_sgd_dmd(20, HIDDEN)
    noprop_dmd = measure_noprop_dmd(20, HIDDEN)
    ff_dmd = measure_ff_dmd(20, HIDDEN)
    all_results['dmd'] = {'sgd': sgd_dmd, 'noprop': noprop_dmd, 'ff': ff_dmd}

    sgd_ard = sgd_dmd['weighted_ard']
    noprop_ard = noprop_dmd['weighted_ard']
    ff_ard = ff_dmd['weighted_ard']
    print(f"  SGD per-step ARD:    {sgd_ard:>12,.1f}")
    print(f"  NoProp per-step ARD: {noprop_ard:>12,.1f}  ({noprop_ard/sgd_ard:.2f}x SGD)")
    print(f"  FF per-step ARD:     {ff_ard:>12,.1f}  ({ff_ard/sgd_ard:.2f}x SGD)")
    print(f"  Note: NoProp does {N_LAYERS} layer-passes/epoch; FF does 2 passes (pos+neg)/step")

    # --- Summary ---
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"  {'Regime':<12} {'Method':<45} {'Solve%':>7} {'Epochs':>8} {'Time':>9}")
    print(f"  {'─'*12} {'─'*45} {'─'*7} {'─'*8} {'─'*9}")

    for regime_name, regime_results in all_results.items():
        if regime_name == 'dmd':
            continue
        for method_name, r in regime_results.items():
            print(f"  {regime_name:<12} {r['label']:<45} {r['solve_rate']:>6.0%} "
                  f"{r['avg_epochs']:>8.0f} {r['avg_time']:>8.3f}s")
        print()

    # --- Save ---
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=to_serializable)
    print(f"  Saved: {RESULTS_DIR / 'results.json'}")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()

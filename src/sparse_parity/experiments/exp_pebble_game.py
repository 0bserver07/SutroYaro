#!/usr/bin/env python3
"""
Experiment exp_pebble_game: Pebble Game Optimizer for Sparse Parity

Hypothesis: By modeling a single training step as a pebble game on a DAG
and optimizing the execution order (topological sort), we can reduce total
energy cost compared to the standard forward-then-backward ordering.

The computation graph has ~15 operations. We enumerate valid topological
orderings (or sample them), simulate memory access patterns, and compute
energy using a tiered memory model:
  register (<=64 floats): 5 pJ/float
  L1 cache (<=64K floats): 20 pJ/float
  L2 cache (<=256K floats): 100 pJ/float
  HBM: 640 pJ/float

Answers: Can execution reordering reduce energy without changing the
learning algorithm (SGD)?

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_pebble_game.py
"""

import time
import json
import math
import random
import itertools
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params, forward, forward_batch
from sparse_parity.tracker import MemTracker
from sparse_parity.metrics import hinge_loss, accuracy

# =============================================================================
# CONFIG
# =============================================================================

EXP_NAME = "exp_pebble_game"
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / EXP_NAME

CONFIG = Config(
    n_bits=20,
    k_sparse=3,
    hidden=1000,
    lr=0.1,
    wd=0.01,
    max_epochs=30,  # converges in ~10 epochs; 30 for safety margin
    n_train=500,
    n_test=200,
    seed=42,
)

# Energy costs per float access (picojoules)
ENERGY_REGISTER = 5     # register file, <= 64 floats
ENERGY_L1 = 20          # L1 cache, <= 64K floats
ENERGY_L2 = 100         # L2 cache, <= 256K floats
ENERGY_HBM = 640        # HBM / main memory

# Cache capacities in floats (float32)
L1_CAPACITY = 64 * 1024 // 4   # 64 KB = 16384 floats
L2_CAPACITY = 256 * 1024 // 4  # 256 KB = 65536 floats
REG_CAPACITY = 64               # 64 registers

# =============================================================================
# COMPUTATION GRAPH (DAG) FOR ONE TRAINING STEP
# =============================================================================

def build_computation_graph(n_bits, hidden):
    """
    Build the DAG for one training step of a 2-layer MLP with hinge loss.

    Each node is an operation that produces a tensor.
    Edges encode data dependencies (which tensors must be alive).

    Returns:
        nodes: list of (name, output_size, input_names)
            - name: operation identifier
            - output_size: number of floats produced
            - input_names: list of tensor names this op reads
    """
    H = hidden
    N = n_bits

    # The "persistent" tensors (parameters + data) are pre-loaded.
    # We model them as initial nodes with no dependencies.
    initial_tensors = {
        'W1': H * N,
        'b1': H,
        'W2': H,
        'b2': 1,
        'x': N,
        'y': 1,
    }

    # Operations in the computation graph.
    # Each tuple: (op_name, output_tensor_name, output_size, [input_tensor_names])
    ops = [
        # Forward pass
        ('fwd_linear1', 'h_pre', H, ['W1', 'x', 'b1']),
        ('fwd_relu', 'h', H, ['h_pre']),
        ('fwd_linear2', 'y_hat', 1, ['W2', 'h', 'b2']),
        ('fwd_loss', 'loss', 1, ['y_hat', 'y']),

        # Backward pass
        ('bwd_loss', 'dy', 1, ['loss', 'y_hat', 'y']),
        ('bwd_dW2', 'dW2', H, ['dy', 'h']),
        ('bwd_db2', 'db2', 1, ['dy']),
        ('bwd_dh', 'dh', H, ['W2', 'dy']),
        ('bwd_relu', 'dh_pre', H, ['dh', 'h_pre']),
        ('bwd_dW1', 'dW1', H * N, ['dh_pre', 'x']),
        ('bwd_db1', 'db1', H, ['dh_pre']),

        # Parameter updates
        ('upd_W1', 'W1_new', H * N, ['W1', 'dW1']),
        ('upd_b1', 'b1_new', H, ['b1', 'db1']),
        ('upd_W2', 'W2_new', H, ['W2', 'dW2']),
        ('upd_b2', 'b2_new', 1, ['b2', 'db2']),
    ]

    return initial_tensors, ops


def get_dependency_graph(ops):
    """
    Build a dependency graph: for each op, which ops must come before it?

    An op B depends on op A if B reads a tensor that A produces.
    """
    # Map: tensor_name -> op that produces it
    producer = {}
    for op_name, out_tensor, out_size, inputs in ops:
        producer[out_tensor] = op_name

    # Build dependencies
    deps = {}
    for op_name, out_tensor, out_size, inputs in ops:
        dep_ops = set()
        for inp in inputs:
            if inp in producer:
                dep_ops.add(producer[inp])
            # If inp is an initial tensor (not produced by any op), no dependency
        deps[op_name] = dep_ops

    return deps


def all_topological_sorts(ops):
    """
    Enumerate ALL valid topological orderings of the ops DAG.
    Since the graph has ~15 nodes, this is feasible if the number
    of orderings is manageable (thousands, not billions).

    Uses Kahn's algorithm with backtracking.
    """
    deps = get_dependency_graph(ops)
    op_names = [op[0] for op in ops]
    n = len(op_names)

    # in-degree (only counting op->op dependencies, not initial tensors)
    in_degree = {name: len(deps[name]) for name in op_names}

    # reverse deps: who depends on me?
    rev_deps = defaultdict(list)
    for name in op_names:
        for dep in deps[name]:
            rev_deps[dep].append(name)

    results = []

    def backtrack(order, in_deg):
        if len(order) == n:
            results.append(list(order))
            return

        # Find all ops with in-degree 0 (all dependencies satisfied)
        available = [name for name in op_names
                     if name not in set(order) and in_deg[name] == 0]

        for choice in available:
            order.append(choice)
            # Decrease in-degree for dependents
            new_in_deg = dict(in_deg)
            for dep in rev_deps[choice]:
                new_in_deg[dep] -= 1
            backtrack(order, new_in_deg)
            order.pop()

    backtrack([], dict(in_degree))
    return results


def sample_topological_sorts(ops, n_samples=10000, rng=None):
    """
    Sample random topological orderings using Kahn's algorithm
    with random tie-breaking. Falls back to this if exhaustive
    enumeration produces too many orderings.
    """
    if rng is None:
        rng = random.Random(42)

    deps = get_dependency_graph(ops)
    op_names = [op[0] for op in ops]

    in_degree = {name: len(deps[name]) for name in op_names}

    rev_deps = defaultdict(list)
    for name in op_names:
        for dep in deps[name]:
            rev_deps[dep].append(name)

    results = set()
    for _ in range(n_samples * 3):  # oversample to get unique orderings
        order = []
        in_deg = dict(in_degree)
        remaining = set(op_names)

        while remaining:
            available = [n for n in remaining if in_deg[n] == 0]
            if not available:
                break
            choice = rng.choice(available)
            order.append(choice)
            remaining.remove(choice)
            for dep in rev_deps[choice]:
                in_deg[dep] -= 1

        results.add(tuple(order))
        if len(results) >= n_samples:
            break

    return [list(t) for t in results]


# =============================================================================
# ENERGY SIMULATION
# =============================================================================

def compute_energy_for_ordering(ordering, initial_tensors, ops, verbose=False):
    """
    Simulate executing ops in the given order and compute total energy.

    Memory model:
    - We track a "working set" of live tensors and their sizes.
    - When an op reads a tensor, the energy depends on whether
      the tensor fits in register/L1/L2 or must come from HBM.
    - We use the cumulative working set size to determine which
      tier each tensor is accessed from, using LRU-like heuristic:
      most recently produced/accessed tensors are in faster memory.

    Simplified model: at each operation, we have a set of live tensors.
    We sort them by "last access time" (most recent first). The first
    REG_CAPACITY floats are in registers, next L1_CAPACITY in L1, etc.
    """
    op_map = {op[0]: op for op in ops}
    op_positions = {name: idx for idx, name in enumerate(ordering)}

    # Track tensor metadata
    tensor_size = dict(initial_tensors)  # tensor_name -> size in floats
    for op_name, out_tensor, out_size, inputs in ops:
        tensor_size[out_tensor] = out_size

    # Track which tensors are "alive" and their last access time
    # Initial tensors are all alive at time 0
    last_access = {}
    for t in initial_tensors:
        last_access[t] = 0

    # Track which tensors have been produced
    produced = set(initial_tensors.keys())

    total_energy = 0.0
    energy_breakdown = defaultdict(float)  # per-op energy
    tier_breakdown = defaultdict(float)     # energy per tier
    access_log = []  # for debugging

    time_step = 0

    for op_name in ordering:
        op = op_map[op_name]
        _, out_tensor, out_size, inputs = op
        time_step += 1

        # Determine memory tier for each input tensor
        op_energy = 0.0

        for inp in inputs:
            size = tensor_size[inp]

            # Determine tier based on working set position
            # Sort all live tensors by last access time (most recent first)
            live_tensors = sorted(last_access.items(), key=lambda x: -x[1])

            # Assign tiers: most recently accessed go to fastest memory
            cumulative = 0
            tier = 'HBM'
            for t_name, t_time in live_tensors:
                t_size = tensor_size[t_name]
                if t_name == inp:
                    if cumulative + t_size <= REG_CAPACITY:
                        tier = 'register'
                    elif cumulative + t_size <= REG_CAPACITY + L1_CAPACITY:
                        tier = 'L1'
                    elif cumulative + t_size <= REG_CAPACITY + L1_CAPACITY + L2_CAPACITY:
                        tier = 'L2'
                    else:
                        tier = 'HBM'
                    break
                cumulative += t_size

            # Compute energy for this access
            cost_per_float = {
                'register': ENERGY_REGISTER,
                'L1': ENERGY_L1,
                'L2': ENERGY_L2,
                'HBM': ENERGY_HBM,
            }[tier]

            access_energy = size * cost_per_float
            op_energy += access_energy
            tier_breakdown[tier] += access_energy

            if verbose:
                access_log.append(
                    f"  {op_name}: read {inp} ({size} floats) from {tier} "
                    f"-> {access_energy/1e6:.3f} uJ"
                )

            # Update last access time
            last_access[inp] = time_step

        # Write output tensor
        tensor_size[out_tensor] = out_size
        last_access[out_tensor] = time_step
        produced.add(out_tensor)

        # Energy for writing output (goes to fastest available tier)
        write_tier = 'register' if out_size <= REG_CAPACITY else \
                     'L1' if out_size <= L1_CAPACITY else \
                     'L2' if out_size <= L2_CAPACITY else 'HBM'
        write_cost = {
            'register': ENERGY_REGISTER,
            'L1': ENERGY_L1,
            'L2': ENERGY_L2,
            'HBM': ENERGY_HBM,
        }[write_tier]
        write_energy = out_size * write_cost
        op_energy += write_energy
        tier_breakdown[write_tier] += write_energy

        if verbose:
            access_log.append(
                f"  {op_name}: write {out_tensor} ({out_size} floats) to {write_tier} "
                f"-> {write_energy/1e6:.3f} uJ"
            )

        energy_breakdown[op_name] = op_energy
        total_energy += op_energy

        # Garbage collect: remove tensors no longer needed
        # A tensor is dead if no future op reads it
        op_idx = op_positions[op_name]
        remaining_ops = ordering[op_idx + 1:]
        future_reads = set()
        for future_op_name in remaining_ops:
            future_op = op_map[future_op_name]
            for inp in future_op[3]:
                future_reads.add(inp)

        dead = [t for t in list(last_access.keys())
                if t not in future_reads and t != out_tensor
                and t not in initial_tensors]  # keep params alive
        for t in dead:
            del last_access[t]

    return {
        'total_energy_pJ': total_energy,
        'total_energy_uJ': total_energy / 1e6,
        'energy_breakdown': dict(energy_breakdown),
        'tier_breakdown': dict(tier_breakdown),
        'access_log': access_log if verbose else None,
        'ordering': ordering,
    }


# =============================================================================
# PREDEFINED ORDERINGS (STRATEGIES)
# =============================================================================

def standard_ordering():
    """Standard forward-then-backward order."""
    return [
        'fwd_linear1', 'fwd_relu', 'fwd_linear2', 'fwd_loss',
        'bwd_loss', 'bwd_dW2', 'bwd_db2', 'bwd_dh', 'bwd_relu',
        'bwd_dW1', 'bwd_db1',
        'upd_W1', 'upd_b1', 'upd_W2', 'upd_b2',
    ]


def fused_ordering():
    """Fused: compute gradients and update immediately per layer (backward)."""
    return [
        'fwd_linear1', 'fwd_relu', 'fwd_linear2', 'fwd_loss',
        'bwd_loss',
        'bwd_dW2', 'upd_W2',  # fuse layer 2 grad + update
        'bwd_db2', 'upd_b2',
        'bwd_dh', 'bwd_relu',
        'bwd_dW1', 'upd_W1',  # fuse layer 1 grad + update
        'bwd_db1', 'upd_b1',
    ]


def perlayer_ordering():
    """Per-layer: process each layer fully before moving to the next."""
    return [
        'fwd_linear1', 'fwd_relu', 'fwd_linear2', 'fwd_loss',
        'bwd_loss',
        'bwd_dW2', 'bwd_db2', 'upd_W2', 'upd_b2',  # layer 2 complete
        'bwd_dh', 'bwd_relu',
        'bwd_dW1', 'bwd_db1', 'upd_W1', 'upd_b1',  # layer 1 complete
    ]


def is_valid_topological_order(ordering, ops):
    """Check if an ordering respects all dependencies."""
    deps = get_dependency_graph(ops)
    completed = set()
    for op_name in ordering:
        if not deps[op_name].issubset(completed):
            return False
        completed.add(op_name)
    return True


# =============================================================================
# ACTUAL TRAINING WITH DIFFERENT ORDERINGS
# =============================================================================

def train_one_step_reordered(x, y, W1, b1, W2, b2, config, ordering, tracker=None):
    """
    Execute one training step following the given operation ordering.
    This validates that reordering does not change the mathematical result
    (since SGD is fixed, only execution order changes).

    In practice, for a 2-layer net, the forward pass must happen before
    backward (data dependency), but within the backward pass and updates,
    the order of independent operations can change.
    """
    hidden = len(W1)
    n_bits = len(x)

    # Storage for intermediate results
    tensors = {
        'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
        'x': x, 'y': y,
    }

    if tracker:
        tracker.write('W1', hidden * n_bits)
        tracker.write('b1', hidden)
        tracker.write('W2', hidden)
        tracker.write('b2', 1)
        tracker.write('x', n_bits)
        tracker.write('y', 1)

    for op_name in ordering:
        if op_name == 'fwd_linear1':
            if tracker:
                tracker.read('W1', hidden * n_bits)
                tracker.read('x', n_bits)
                tracker.read('b1', hidden)
            h_pre = [sum(W1[j][i] * x[i] for i in range(n_bits)) + b1[j]
                     for j in range(hidden)]
            tensors['h_pre'] = h_pre
            if tracker:
                tracker.write('h_pre', hidden)

        elif op_name == 'fwd_relu':
            if tracker:
                tracker.read('h_pre', hidden)
            h = [max(0.0, v) for v in tensors['h_pre']]
            tensors['h'] = h
            if tracker:
                tracker.write('h', hidden)

        elif op_name == 'fwd_linear2':
            if tracker:
                tracker.read('W2', hidden)
                tracker.read('h', hidden)
                tracker.read('b2', 1)
            h = tensors['h']
            out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]
            tensors['y_hat'] = out
            if tracker:
                tracker.write('y_hat', 1)

        elif op_name == 'fwd_loss':
            if tracker:
                tracker.read('y_hat', 1)
                tracker.read('y', 1)
            margin = tensors['y_hat'] * y
            tensors['loss'] = max(0.0, 1.0 - margin)
            tensors['margin'] = margin
            if tracker:
                tracker.write('loss', 1)

        elif op_name == 'bwd_loss':
            if tracker:
                tracker.read('loss', 1)
                tracker.read('y_hat', 1)
                tracker.read('y', 1)
            margin = tensors.get('margin', tensors['y_hat'] * y)
            if margin >= 1.0:
                tensors['dy'] = 0.0
                tensors['_skip_grad'] = True
            else:
                tensors['dy'] = -y
                tensors['_skip_grad'] = False
            if tracker:
                tracker.write('dy', 1)

        elif op_name == 'bwd_dW2':
            if tracker:
                tracker.read('dy', 1)
                tracker.read('h', hidden)
            dy = tensors['dy']
            h = tensors['h']
            dW2 = [dy * h[j] for j in range(hidden)]
            tensors['dW2'] = dW2
            if tracker:
                tracker.write('dW2', hidden)

        elif op_name == 'bwd_db2':
            if tracker:
                tracker.read('dy', 1)
            tensors['db2'] = tensors['dy']
            if tracker:
                tracker.write('db2', 1)

        elif op_name == 'bwd_dh':
            if tracker:
                tracker.read('W2', hidden)
                tracker.read('dy', 1)
            dy = tensors['dy']
            dh = [W2[0][j] * dy for j in range(hidden)]
            tensors['dh'] = dh
            if tracker:
                tracker.write('dh', hidden)

        elif op_name == 'bwd_relu':
            if tracker:
                tracker.read('dh', hidden)
                tracker.read('h_pre', hidden)
            dh = tensors['dh']
            h_pre = tensors['h_pre']
            dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0)
                      for j in range(hidden)]
            tensors['dh_pre'] = dh_pre
            if tracker:
                tracker.write('dh_pre', hidden)

        elif op_name == 'bwd_dW1':
            if tracker:
                tracker.read('dh_pre', hidden)
                tracker.read('x', n_bits)
            dh_pre = tensors['dh_pre']
            dW1 = [[dh_pre[j] * x[i] for i in range(n_bits)]
                    for j in range(hidden)]
            tensors['dW1'] = dW1
            if tracker:
                tracker.write('dW1', hidden * n_bits)

        elif op_name == 'bwd_db1':
            if tracker:
                tracker.read('dh_pre', hidden)
            tensors['db1'] = list(tensors['dh_pre'])
            if tracker:
                tracker.write('db1', hidden)

        elif op_name == 'upd_W1':
            if tracker:
                tracker.read('W1', hidden * n_bits)
                tracker.read('dW1', hidden * n_bits)
            dW1 = tensors['dW1']
            for j in range(hidden):
                for i in range(n_bits):
                    grad = dW1[j][i]
                    W1[j][i] -= config.lr * (grad + config.wd * W1[j][i])
            if tracker:
                tracker.write('W1', hidden * n_bits)

        elif op_name == 'upd_b1':
            if tracker:
                tracker.read('b1', hidden)
                tracker.read('db1', hidden)
            db1_val = tensors['db1']
            for j in range(hidden):
                b1[j] -= config.lr * (db1_val[j] + config.wd * b1[j])
            if tracker:
                tracker.write('b1', hidden)

        elif op_name == 'upd_W2':
            if tracker:
                tracker.read('W2', hidden)
                tracker.read('dW2', hidden)
            dW2 = tensors['dW2']
            for j in range(hidden):
                W2[0][j] -= config.lr * (dW2[j] + config.wd * W2[0][j])
            if tracker:
                tracker.write('W2', hidden)

        elif op_name == 'upd_b2':
            if tracker:
                tracker.read('b2', 1)
                tracker.read('db2', 1)
            db2_val = tensors['db2']
            b2[0] -= config.lr * (db2_val + config.wd * b2[0])
            if tracker:
                tracker.write('b2', 1)

    return tensors.get('y_hat', 0.0), tensors.get('h_pre'), tensors.get('h')


def train_with_ordering(x_train, y_train, x_test, y_test,
                        W1, b1, W2, b2, config, ordering, tracker_step=0):
    """
    Full training loop using a specific operation ordering per step.
    """
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    step = 0
    best_test_acc = 0.0
    tracker_result = None

    start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        for i in range(len(x_train)):
            tracker = MemTracker() if step == tracker_step else None

            train_one_step_reordered(
                x_train[i], y_train[i], W1, b1, W2, b2,
                config, ordering, tracker=tracker
            )

            if tracker:
                tracker_result = tracker.to_json()

            step += 1

        # Evaluate after each epoch
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
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc,
        'total_steps': step,
        'elapsed_s': elapsed,
        'tracker': tracker_result,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    sys.stdout.reconfigure(line_buffering=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'='*70}")
    print(f"  EXPERIMENT: {EXP_NAME}")
    print(f"  Pebble Game Optimizer for Sparse Parity Training")
    print(f"  Config: n={CONFIG.n_bits}, k={CONFIG.k_sparse}, hidden={CONFIG.hidden}")
    print(f"{'='*70}")

    # --- Phase 1: Build computation graph ---
    print("\n  [Phase 1] Building computation graph...")
    initial_tensors, ops = build_computation_graph(CONFIG.n_bits, CONFIG.hidden)

    print(f"    Initial tensors: {len(initial_tensors)}")
    for name, size in initial_tensors.items():
        print(f"      {name}: {size:,} floats")

    print(f"    Operations: {len(ops)}")
    for op_name, out_tensor, out_size, inputs in ops:
        print(f"      {op_name}: {inputs} -> {out_tensor} ({out_size:,} floats)")

    # --- Phase 2: Sample topological orderings ---
    print("\n  [Phase 2] Sampling topological orderings...")
    print(f"    (Exhaustive enumeration infeasible: 15-node DAG has millions of valid orderings)")

    deps = get_dependency_graph(ops)
    print(f"    Dependency graph:")
    for op, d in deps.items():
        print(f"      {op} depends on: {d if d else '{}'}")

    N_SAMPLES = 10000
    t0 = time.time()
    orderings = sample_topological_sorts(ops, n_samples=N_SAMPLES)
    exhaustive = False
    print(f"    Sampled {len(orderings)} unique orderings in {time.time()-t0:.2f}s")

    # --- Phase 3: Evaluate energy for all orderings ---
    print("\n  [Phase 3] Evaluating energy for each ordering...")

    # Evaluate predefined strategies first
    named_strategies = {
        'standard': standard_ordering(),
        'fused': fused_ordering(),
        'perlayer': perlayer_ordering(),
    }

    # Validate predefined strategies
    for name, order in named_strategies.items():
        valid = is_valid_topological_order(order, ops)
        print(f"    {name} ordering valid: {valid}")
        if not valid:
            print(f"      WARNING: {name} ordering is not a valid topological sort!")

    strategy_results = {}
    for name, order in named_strategies.items():
        if is_valid_topological_order(order, ops):
            result = compute_energy_for_ordering(order, initial_tensors, ops, verbose=True)
            strategy_results[name] = result
            print(f"    {name}: {result['total_energy_uJ']:.2f} uJ")

    # Evaluate all enumerated/sampled orderings
    t0 = time.time()
    energies = []
    for i, order in enumerate(orderings):
        result = compute_energy_for_ordering(order, initial_tensors, ops)
        energies.append((result['total_energy_pJ'], i, order))

    energies.sort()
    eval_time = time.time() - t0
    print(f"    Evaluated {len(energies)} orderings in {eval_time:.2f}s")

    # Best and worst
    best_energy, best_idx, best_order = energies[0]
    worst_energy, worst_idx, worst_order = energies[-1]

    best_result = compute_energy_for_ordering(best_order, initial_tensors, ops, verbose=True)
    worst_result = compute_energy_for_ordering(worst_order, initial_tensors, ops, verbose=True)

    print(f"\n    Best ordering:  {best_result['total_energy_uJ']:.2f} uJ")
    print(f"    Worst ordering: {worst_result['total_energy_uJ']:.2f} uJ")
    print(f"    Ratio (worst/best): {worst_energy/best_energy:.2f}x")

    # Check if standard is among evaluated
    std_energy = strategy_results.get('standard', {}).get('total_energy_pJ', 0)
    if std_energy > 0:
        print(f"    Standard: {std_energy/1e6:.2f} uJ")
        print(f"    Improvement (best vs standard): "
              f"{(1 - best_energy/std_energy)*100:.1f}%")

    # Print the best ordering
    print(f"\n    Best ordering:")
    for j, op in enumerate(best_order):
        print(f"      {j+1}. {op}")

    # Tier breakdown for best ordering
    print(f"\n    Energy breakdown by tier (best ordering):")
    for tier, energy in sorted(best_result['tier_breakdown'].items()):
        print(f"      {tier}: {energy/1e6:.2f} uJ ({energy/best_energy*100:.1f}%)")

    # Tier breakdown for standard ordering
    if 'standard' in strategy_results:
        print(f"\n    Energy breakdown by tier (standard ordering):")
        for tier, energy in sorted(strategy_results['standard']['tier_breakdown'].items()):
            total_std = strategy_results['standard']['total_energy_pJ']
            print(f"      {tier}: {energy/1e6:.2f} uJ ({energy/total_std*100:.1f}%)")

    # --- Phase 4: Run actual training with best vs standard ordering ---
    print(f"\n  [Phase 4] Validating with actual training...")

    data = generate(CONFIG)
    x_train, y_train, x_test, y_test, secret = data
    print(f"    Secret indices: {secret}")

    # Standard ordering training
    print(f"\n    Training with STANDARD ordering...")
    W1s, b1s, W2s, b2s = init_params(CONFIG)
    t0 = time.time()
    std_result = train_with_ordering(
        x_train, y_train, x_test, y_test,
        W1s, b1s, W2s, b2s, CONFIG,
        standard_ordering(), tracker_step=0
    )
    std_time = time.time() - t0
    print(f"      Accuracy: {std_result['best_test_acc']:.0%} in {std_time:.1f}s")
    if std_result['tracker']:
        print(f"      ARD: {std_result['tracker']['weighted_ard']:,.0f}")

    # Fused ordering training
    print(f"\n    Training with FUSED ordering...")
    W1f, b1f, W2f, b2f = init_params(CONFIG)
    t0 = time.time()
    fused_result = train_with_ordering(
        x_train, y_train, x_test, y_test,
        W1f, b1f, W2f, b2f, CONFIG,
        fused_ordering(), tracker_step=0
    )
    fused_time = time.time() - t0
    print(f"      Accuracy: {fused_result['best_test_acc']:.0%} in {fused_time:.1f}s")
    if fused_result['tracker']:
        print(f"      ARD: {fused_result['tracker']['weighted_ard']:,.0f}")

    # Per-layer ordering training
    print(f"\n    Training with PERLAYER ordering...")
    W1p, b1p, W2p, b2p = init_params(CONFIG)
    t0 = time.time()
    perlayer_result = train_with_ordering(
        x_train, y_train, x_test, y_test,
        W1p, b1p, W2p, b2p, CONFIG,
        perlayer_ordering(), tracker_step=0
    )
    perlayer_time = time.time() - t0
    print(f"      Accuracy: {perlayer_result['best_test_acc']:.0%} in {perlayer_time:.1f}s")
    if perlayer_result['tracker']:
        print(f"      ARD: {perlayer_result['tracker']['weighted_ard']:,.0f}")

    # Best (optimal pebbling) ordering training
    print(f"\n    Training with OPTIMAL PEBBLING ordering...")
    W1o, b1o, W2o, b2o = init_params(CONFIG)
    t0 = time.time()
    optimal_result = train_with_ordering(
        x_train, y_train, x_test, y_test,
        W1o, b1o, W2o, b2o, CONFIG,
        best_order, tracker_step=0
    )
    optimal_time = time.time() - t0
    print(f"      Accuracy: {optimal_result['best_test_acc']:.0%} in {optimal_time:.1f}s")
    if optimal_result['tracker']:
        print(f"      ARD: {optimal_result['tracker']['weighted_ard']:,.0f}")

    # --- Phase 5: Summary ---
    print(f"\n  {'='*70}")
    print(f"  SUMMARY")
    print(f"  {'='*70}")

    all_results = {
        'standard': (strategy_results.get('standard', {}), std_result),
        'fused': (strategy_results.get('fused', {}), fused_result),
        'perlayer': (strategy_results.get('perlayer', {}), perlayer_result),
        'optimal_pebble': (best_result, optimal_result),
    }

    print(f"\n  {'Strategy':<20} {'Energy (uJ)':>12} {'Accuracy':>10} {'ARD':>12} {'Epochs':>8}")
    print(f"  {'─'*20} {'─'*12} {'─'*10} {'─'*12} {'─'*8}")

    for name, (energy_info, train_info) in all_results.items():
        energy_uJ = energy_info.get('total_energy_uJ', 0)
        acc = train_info['best_test_acc']
        ard = train_info['tracker']['weighted_ard'] if train_info.get('tracker') else 'N/A'
        epochs = len(train_info.get('test_accs', []))
        ard_str = f"{ard:>12,.0f}" if isinstance(ard, (int, float)) else f"{'N/A':>12}"
        print(f"  {name:<20} {energy_uJ:>12.2f} {acc:>9.0%} {ard_str} {epochs:>8}")

    # Energy comparison
    std_e = strategy_results.get('standard', {}).get('total_energy_pJ', 1)
    print(f"\n  Energy relative to standard:")
    for name, (energy_info, _) in all_results.items():
        e = energy_info.get('total_energy_pJ', 0)
        if std_e > 0 and e > 0:
            print(f"    {name}: {e/std_e:.4f}x ({(1 - e/std_e)*100:+.1f}%)")

    # Distribution statistics
    all_energies_pJ = [e[0] for e in energies]
    mean_e = sum(all_energies_pJ) / len(all_energies_pJ)
    var_e = sum((e - mean_e)**2 for e in all_energies_pJ) / len(all_energies_pJ)
    std_dev = math.sqrt(var_e)
    median_e = sorted(all_energies_pJ)[len(all_energies_pJ) // 2]

    print(f"\n  Energy distribution over {len(energies)} orderings:")
    print(f"    Min:    {min(all_energies_pJ)/1e6:.2f} uJ")
    print(f"    Max:    {max(all_energies_pJ)/1e6:.2f} uJ")
    print(f"    Mean:   {mean_e/1e6:.2f} uJ")
    print(f"    Median: {median_e/1e6:.2f} uJ")
    print(f"    Std:    {std_dev/1e6:.2f} uJ")
    print(f"    Exhaustive: {exhaustive}")

    # --- Save results ---
    results = {
        'experiment': EXP_NAME,
        'config': {k: v for k, v in CONFIG.__dict__.items()},
        'energy_model': {
            'register_pJ': ENERGY_REGISTER,
            'L1_pJ': ENERGY_L1,
            'L2_pJ': ENERGY_L2,
            'HBM_pJ': ENERGY_HBM,
            'L1_capacity_floats': L1_CAPACITY,
            'L2_capacity_floats': L2_CAPACITY,
            'register_capacity_floats': REG_CAPACITY,
        },
        'graph': {
            'n_ops': len(ops),
            'n_initial_tensors': len(initial_tensors),
            'initial_tensor_sizes': initial_tensors,
        },
        'enumeration': {
            'exhaustive': exhaustive,
            'n_orderings_evaluated': len(energies),
        },
        'energy_distribution': {
            'min_uJ': min(all_energies_pJ) / 1e6,
            'max_uJ': max(all_energies_pJ) / 1e6,
            'mean_uJ': mean_e / 1e6,
            'median_uJ': median_e / 1e6,
            'std_uJ': std_dev / 1e6,
        },
        'strategies': {},
        'best_ordering': best_order,
        'training_results': {},
    }

    for name, (energy_info, train_info) in all_results.items():
        results['strategies'][name] = {
            'energy_uJ': energy_info.get('total_energy_uJ', 0),
            'energy_pJ': energy_info.get('total_energy_pJ', 0),
            'tier_breakdown': energy_info.get('tier_breakdown', {}),
            'ordering': energy_info.get('ordering', []),
        }
        results['training_results'][name] = {
            'best_test_acc': train_info['best_test_acc'],
            'total_steps': train_info['total_steps'],
            'elapsed_s': train_info['elapsed_s'],
            'epochs': len(train_info.get('test_accs', [])),
            'tracker_ard': (train_info['tracker']['weighted_ard']
                           if train_info.get('tracker') else None),
        }

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_DIR / 'results.json'}")


if __name__ == '__main__':
    main()

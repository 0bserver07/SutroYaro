#!/usr/bin/env python3
"""
Experiment: Genetic Programming for Sparse Parity

Hypothesis: GP can evolve expression trees that compute sparse parity.
The ideal solution is sign(x[a] * x[b] * x[c]) -- a depth-3 tree with
zero learned parameters, zero memory footprint, zero ARD.

Approach #14: Evolve symbolic expression trees using crossover and mutation.
Primitives: multiply, negate, sign. Terminals: x[i], 1.0, -1.0.

Key insight: since inputs are {-1, +1}, sign is idempotent and negate
just flips the sign. The effective search is for products of k variables.
We use parsimony pressure and validation to prevent overfitting.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_genetic_prog.py
"""

import sys
import time
import json
import copy
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sparse_parity.tracker import MemTracker


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n_bits, k_sparse, n_samples, seed=42, secret=None):
    """Generate sparse parity data. Returns x, y, secret.
    If secret is provided, use it instead of generating one from seed."""
    rng = np.random.RandomState(seed)
    if secret is None:
        secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    return x, y, secret


# =============================================================================
# EXPRESSION TREE REPRESENTATION
# =============================================================================

# Tree nodes are tuples:
#   Terminal: ('var', i)       -- x[i]
#             ('const', val)   -- 1.0 or -1.0
#   Function: ('mul', left, right)
#             ('neg', child)
#             ('sign', child)

def tree_depth(node):
    """Return the depth of a tree."""
    if node[0] in ('var', 'const'):
        return 1
    elif node[0] in ('neg', 'sign'):
        return 1 + tree_depth(node[1])
    elif node[0] == 'mul':
        return 1 + max(tree_depth(node[1]), tree_depth(node[2]))
    return 1


def tree_size(node):
    """Return number of nodes in the tree."""
    if node[0] in ('var', 'const'):
        return 1
    elif node[0] in ('neg', 'sign'):
        return 1 + tree_size(node[1])
    elif node[0] == 'mul':
        return 1 + tree_size(node[1]) + tree_size(node[2])
    return 1


def eval_tree_batch(node, x):
    """Evaluate tree on all rows of x. Returns array of predictions."""
    if node[0] == 'var':
        return x[:, node[1]].copy()
    elif node[0] == 'const':
        return np.full(x.shape[0], node[1])
    elif node[0] == 'mul':
        return eval_tree_batch(node[1], x) * eval_tree_batch(node[2], x)
    elif node[0] == 'neg':
        return -eval_tree_batch(node[1], x)
    elif node[0] == 'sign':
        return np.sign(eval_tree_batch(node[1], x))
    return np.zeros(x.shape[0])


def tree_to_str(node):
    """Pretty print a tree as a string."""
    if node[0] == 'var':
        return f"x[{node[1]}]"
    elif node[0] == 'const':
        return str(node[1])
    elif node[0] == 'mul':
        return f"mul({tree_to_str(node[1])}, {tree_to_str(node[2])})"
    elif node[0] == 'neg':
        return f"neg({tree_to_str(node[1])})"
    elif node[0] == 'sign':
        return f"sign({tree_to_str(node[1])})"
    return "?"


def get_variables_used(node):
    """Return set of variable indices used in the tree."""
    if node[0] == 'var':
        return {node[1]}
    elif node[0] == 'const':
        return set()
    elif node[0] in ('neg', 'sign'):
        return get_variables_used(node[1])
    elif node[0] == 'mul':
        return get_variables_used(node[1]) | get_variables_used(node[2])
    return set()


# =============================================================================
# RANDOM TREE GENERATION
# =============================================================================

def random_tree(n_bits, max_depth, rng, method='grow'):
    """
    Generate a random expression tree.
    method='full': always pick functions until max_depth reached
    method='grow': randomly pick functions or terminals
    """
    if max_depth <= 1:
        # Must be a terminal
        if rng.random() < 0.9:
            return ('var', int(rng.randint(n_bits)))
        else:
            return ('const', rng.choice([1.0, -1.0]))

    if method == 'full' or (method == 'grow' and rng.random() < 0.65):
        # Pick a function -- bias heavily toward multiply
        func = rng.choice(['mul', 'neg', 'sign'], p=[0.7, 0.1, 0.2])
        if func == 'mul':
            left = random_tree(n_bits, max_depth - 1, rng, method)
            right = random_tree(n_bits, max_depth - 1, rng, method)
            return ('mul', left, right)
        elif func == 'neg':
            child = random_tree(n_bits, max_depth - 1, rng, method)
            return ('neg', child)
        else:
            child = random_tree(n_bits, max_depth - 1, rng, method)
            return ('sign', child)
    else:
        if rng.random() < 0.9:
            return ('var', int(rng.randint(n_bits)))
        else:
            return ('const', rng.choice([1.0, -1.0]))


# =============================================================================
# GP OPERATIONS
# =============================================================================

def get_all_nodes(node, path=()):
    """Return list of (path, subnode) for all nodes in the tree."""
    result = [(path, node)]
    if node[0] in ('neg', 'sign'):
        result.extend(get_all_nodes(node[1], path + (1,)))
    elif node[0] == 'mul':
        result.extend(get_all_nodes(node[1], path + (1,)))
        result.extend(get_all_nodes(node[2], path + (2,)))
    return result


def replace_at_path(node, path, new_subtree):
    """Return a new tree with the subtree at `path` replaced."""
    if len(path) == 0:
        return new_subtree
    idx = path[0]
    rest = path[1:]
    if node[0] in ('neg', 'sign') and idx == 1:
        return (node[0], replace_at_path(node[1], rest, new_subtree))
    elif node[0] == 'mul':
        if idx == 1:
            return ('mul', replace_at_path(node[1], rest, new_subtree), node[2])
        elif idx == 2:
            return ('mul', node[1], replace_at_path(node[2], rest, new_subtree))
    return node


def crossover(parent_a, parent_b, max_depth, rng):
    """Swap random subtrees between two parents."""
    nodes_a = get_all_nodes(parent_a)
    nodes_b = get_all_nodes(parent_b)
    path_a, _ = nodes_a[rng.randint(len(nodes_a))]
    _, subtree_b = nodes_b[rng.randint(len(nodes_b))]
    child = replace_at_path(parent_a, path_a, subtree_b)
    if tree_depth(child) > max_depth:
        return copy.deepcopy(parent_a)
    return child


def mutation(tree, n_bits, max_depth, rng):
    """Replace a random subtree with a new random subtree."""
    nodes = get_all_nodes(tree)
    path, _ = nodes[rng.randint(len(nodes))]
    depth_budget = max(1, max_depth - len(path)) if len(path) > 0 else max_depth
    new_subtree = random_tree(n_bits, min(depth_budget, 3), rng, method='grow')
    result = replace_at_path(tree, path, new_subtree)
    if tree_depth(result) > max_depth:
        return tree
    return result


def point_mutation(tree, n_bits, rng):
    """Change a single node's type or terminal value."""
    nodes = get_all_nodes(tree)
    path, node = nodes[rng.randint(len(nodes))]
    if node[0] == 'var':
        new_node = ('var', int(rng.randint(n_bits)))
    elif node[0] == 'const':
        new_node = ('const', rng.choice([1.0, -1.0]))
    elif node[0] in ('neg', 'sign'):
        new_type = 'sign' if node[0] == 'neg' else 'neg'
        new_node = (new_type, node[1])
    else:
        return tree
    return replace_at_path(tree, path, new_node)


def tournament_select(population, fitnesses, tournament_size, rng):
    """Pick tournament_size candidates, return the best."""
    indices = rng.choice(len(population), min(tournament_size, len(population)),
                         replace=False)
    best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best_idx]


# =============================================================================
# FITNESS WITH PARSIMONY PRESSURE
# =============================================================================

def compute_accuracy(tree, x, y):
    """Raw classification accuracy of tree on (x, y) data."""
    preds = eval_tree_batch(tree, x)
    pred_labels = np.sign(preds)
    return float(np.mean(pred_labels == y))


def compute_fitness(tree, x, y, parsimony_coeff=0.002):
    """
    Fitness = accuracy - parsimony_coeff * tree_size.
    Parsimony pressure prevents bloat and overfitting.
    """
    acc = compute_accuracy(tree, x, y)
    size = tree_size(tree)
    return acc - parsimony_coeff * size


# =============================================================================
# GP ENGINE
# =============================================================================

def gp_search(x_train, y_train, x_val, y_val, n_bits, k_sparse,
              pop_size=200, max_generations=200,
              max_depth=7, tournament_size=4, crossover_rate=0.7,
              mutation_rate=0.2, point_mutation_rate=0.1,
              parsimony_coeff=0.002,
              seed=42, verbose=True):
    """
    Genetic Programming search for sparse parity.
    Uses train data for fitness, validation data to check generalization.
    Returns (best_tree, generation, elapsed_s, best_train_acc, best_val_acc, stats).
    """
    rng = np.random.RandomState(seed)
    start = time.time()

    # Ramped half-and-half initialization
    population = []
    for i in range(pop_size):
        depth = rng.randint(2, max_depth + 1)
        method = 'full' if i % 2 == 0 else 'grow'
        tree = random_tree(n_bits, depth, rng, method)
        population.append(tree)

    best_ever = None
    best_ever_val_acc = 0.0
    best_ever_train_acc = 0.0
    stats = {'diversity': [], 'best_fitness': [], 'avg_fitness': [],
             'best_val_acc': []}

    for gen in range(1, max_generations + 1):
        # Evaluate fitness (train accuracy with parsimony)
        fitnesses = [compute_fitness(t, x_train, y_train, parsimony_coeff)
                     for t in population]
        fitnesses_arr = np.array(fitnesses)

        # Also compute raw train accuracy for reporting
        train_accs = [compute_accuracy(t, x_train, y_train)
                      for t in population]
        train_accs_arr = np.array(train_accs)

        best_idx = np.argmax(fitnesses_arr)
        gen_best_fitness = fitnesses_arr[best_idx]
        gen_best_train = train_accs_arr[best_idx]
        gen_avg = float(np.mean(fitnesses_arr))

        # Validate the best individual
        val_acc = compute_accuracy(population[best_idx], x_val, y_val)

        # Track best by validation accuracy (must also have high train acc)
        if val_acc > best_ever_val_acc and gen_best_train > 0.9:
            best_ever_val_acc = val_acc
            best_ever_train_acc = gen_best_train
            best_ever = copy.deepcopy(population[best_idx])
        elif best_ever is None:
            best_ever = copy.deepcopy(population[best_idx])
            best_ever_train_acc = gen_best_train
            best_ever_val_acc = val_acc

        # Track diversity
        var_sets = set()
        for t in population:
            vs = frozenset(get_variables_used(t))
            var_sets.add(vs)
        diversity = len(var_sets)

        stats['diversity'].append(diversity)
        stats['best_fitness'].append(float(gen_best_fitness))
        stats['avg_fitness'].append(gen_avg)
        stats['best_val_acc'].append(val_acc)

        if verbose and (gen <= 5 or gen % 50 == 0 or val_acc >= 1.0):
            elapsed = time.time() - start
            print(f"      Gen {gen:>4}: train={gen_best_train:.4f} "
                  f"val={val_acc:.4f} fitness={gen_best_fitness:.4f} "
                  f"div={diversity} t={elapsed:.2f}s")

        # Check for perfect solution on validation
        if val_acc >= 1.0 and gen_best_train >= 1.0:
            elapsed = time.time() - start
            return best_ever, gen, elapsed, best_ever_train_acc, best_ever_val_acc, stats

        # Create next generation
        new_pop = []
        # Elitism: keep top 2
        sorted_idx = np.argsort(fitnesses_arr)[::-1]
        for i in range(min(2, pop_size)):
            new_pop.append(copy.deepcopy(population[sorted_idx[i]]))

        while len(new_pop) < pop_size:
            r = rng.random()
            if r < crossover_rate:
                p1 = tournament_select(population, fitnesses_arr,
                                       tournament_size, rng)
                p2 = tournament_select(population, fitnesses_arr,
                                       tournament_size, rng)
                child = crossover(p1, p2, max_depth, rng)
            elif r < crossover_rate + mutation_rate:
                p1 = tournament_select(population, fitnesses_arr,
                                       tournament_size, rng)
                child = mutation(p1, n_bits, max_depth, rng)
            elif r < crossover_rate + mutation_rate + point_mutation_rate:
                p1 = tournament_select(population, fitnesses_arr,
                                       tournament_size, rng)
                child = point_mutation(p1, n_bits, rng)
            else:
                p1 = tournament_select(population, fitnesses_arr,
                                       tournament_size, rng)
                child = copy.deepcopy(p1)
            new_pop.append(child)

        population = new_pop

    elapsed = time.time() - start
    return best_ever, max_generations, elapsed, best_ever_train_acc, best_ever_val_acc, stats


# =============================================================================
# MEMORY TRACKING
# =============================================================================

def track_memory(tree, x):
    """
    Track memory usage of evaluating the found program.
    The discovered program has zero learned parameters -- only reads input data.
    """
    tracker = MemTracker()
    n_vars = len(get_variables_used(tree))
    n_samples = x.shape[0]

    # The program reads input features (no parameters to store/read)
    tracker.write('input_x', n_samples * n_vars)
    tracker.read('input_x', n_samples * n_vars)

    # Output
    tracker.write('output', n_samples)

    return tracker


# =============================================================================
# RUN CONFIG
# =============================================================================

def run_config(n_bits, k_sparse, n_train, n_test, seeds, pop_size=200,
               max_generations=200, max_depth=7, verbose=True):
    """Run GP on one config across multiple seeds."""
    from math import comb
    c_n_k = comb(n_bits, k_sparse)
    if verbose:
        print(f"\n  Config: n={n_bits}, k={k_sparse}, C(n,k)={c_n_k}")

    results = []

    for seed in seeds:
        if verbose:
            print(f"\n    --- seed={seed} ---")

        # Generate train, validation, and test data (same secret, different x)
        x_train, y_train, secret = generate_data(n_bits, k_sparse, n_train,
                                                  seed=seed)
        x_val, y_val, _ = generate_data(n_bits, k_sparse, n_train,
                                        seed=seed + 500, secret=secret)
        x_test, y_test, _ = generate_data(n_bits, k_sparse, n_test,
                                          seed=seed + 1000, secret=secret)

        # Run GP
        best_tree, gen, elapsed, train_acc, val_acc, stats = gp_search(
            x_train, y_train, x_val, y_val, n_bits, k_sparse,
            pop_size=pop_size, max_generations=max_generations,
            max_depth=max_depth, tournament_size=4,
            seed=seed + 200, verbose=verbose
        )

        # Evaluate on test data
        test_acc = compute_accuracy(best_tree, x_test, y_test)

        # Get variables used
        vars_used = sorted(get_variables_used(best_tree))
        correct_vars = vars_used == secret

        # Memory tracking
        tracker = track_memory(best_tree, x_test)
        mem_summary = tracker.to_json()

        # A true solve requires high val and test accuracy (generalization)
        solved = val_acc >= 0.99 and test_acc >= 0.99
        program_str = tree_to_str(best_tree)
        depth = tree_depth(best_tree)
        size = tree_size(best_tree)

        result = {
            'seed': seed,
            'secret': secret,
            'found_vars': vars_used,
            'correct_vars': correct_vars,
            'solved': solved,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'generations': gen,
            'elapsed_s': round(elapsed, 4),
            'program': program_str,
            'tree_depth': depth,
            'tree_size': size,
            'n_parameters': 0,
            'ard': mem_summary['weighted_ard'],
            'memory': mem_summary,
        }
        results.append(result)

        if verbose:
            status = "SOLVED" if solved else f"FAILED (train={train_acc:.1%}, val={val_acc:.1%}, test={test_acc:.1%})"
            print(f"    Result: {status} in {gen} gens ({elapsed:.3f}s)")
            print(f"    Program: {program_str}")
            print(f"    Vars used: {vars_used}, Secret: {secret}, "
                  f"Correct: {correct_vars}")
            print(f"    Train/Val/Test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}")
            print(f"    Tree depth: {depth}, size: {size}")
            print(f"    Parameters: 0, ARD: {mem_summary['weighted_ard']}")
            tracker.report()

    return {
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'c_n_k': c_n_k,
        'n_train': n_train,
        'n_test': n_test,
        'pop_size': pop_size,
        'max_generations': max_generations,
        'max_depth': max_depth,
        'results': results,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: Genetic Programming for Sparse Parity")
    print("  Approach #14: Evolve expression trees (zero parameters)")
    print("=" * 70)

    seeds = [42, 43, 44]
    all_results = {}

    # -------------------------------------------------------------------
    # Config 1: n=20, k=3  --  C(20,3) = 1,140
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=20, k=3")
    print("=" * 70)
    all_results['n20_k3'] = run_config(
        n_bits=20, k_sparse=3, n_train=1000, n_test=2000, seeds=seeds,
        pop_size=200, max_generations=200, max_depth=6
    )

    # -------------------------------------------------------------------
    # Config 2: n=50, k=3  --  C(50,3) = 19,600
    # Harder: more variables to search through, need more generations
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=50, k=3")
    print("=" * 70)
    all_results['n50_k3'] = run_config(
        n_bits=50, k_sparse=3, n_train=1000, n_test=2000, seeds=seeds,
        pop_size=200, max_generations=500, max_depth=6
    )

    # -------------------------------------------------------------------
    # Config 3: n=20, k=5  --  C(20,5) = 15,504
    # Harder: need deeper trees (5 vars), more generations
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 3: n=20, k=5")
    print("=" * 70)
    all_results['n20_k5'] = run_config(
        n_bits=20, k_sparse=5, n_train=2000, n_test=2000, seeds=seeds,
        pop_size=200, max_generations=500, max_depth=8
    )

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    header = (f"  {'Config':<15} | {'Solved':>8} | {'Avg gens':>9} | "
              f"{'Avg time':>9} | {'Test acc':>9} | {'Params':>7} | {'ARD':>5}")
    print(header)
    print("  " + "-" * 80)

    for key, cfg in all_results.items():
        res_list = cfg['results']
        n_solved = sum(1 for r in res_list if r['solved'])
        total = len(res_list)

        solved_results = [r for r in res_list if r['solved']]
        if solved_results:
            avg_gens = np.mean([r['generations'] for r in solved_results])
            avg_time = np.mean([r['elapsed_s'] for r in solved_results])
            avg_test = np.mean([r['test_acc'] for r in solved_results])
        else:
            avg_gens = float('nan')
            avg_time = float('nan')
            avg_test = np.mean([r['test_acc'] for r in res_list])

        g_str = f"{avg_gens:.0f}" if not np.isnan(avg_gens) else "---"
        t_str = f"{avg_time:.3f}s" if not np.isnan(avg_time) else "---"

        print(f"  {key:<15} | {n_solved}/{total:>5} | {g_str:>9} | "
              f"{t_str:>9} | {avg_test:>8.4f} | {0:>7} | {0:>5}")

    print("=" * 90)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_genetic_prog'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_genetic_prog',
            'approach': '#14 Genetic Programming',
            'description': ('Evolve symbolic expression trees that compute '
                            'sparse parity. Zero parameters, zero ARD.'),
            'configs': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()

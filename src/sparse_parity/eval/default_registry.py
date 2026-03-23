"""
Default challenge and method registrations.

Imported by __init__.py to populate the registry on package load.
This is the single place that maps the 3 current challenges and 16
methods that were previously hardcoded in env.py.

To add a new challenge or method, either:
  1. Add it here (if it ships with the repo), or
  2. Call register_challenge / register_method from your own module
     before creating the Gymnasium environment.
"""

from sparse_parity.eval.registry import register_challenge, register_method


def _import_harness():
    """Lazy import so harness.py (which lives in src/) is only loaded
    when the environment is actually used, not at package-scan time."""
    import harness  # src/ must be on PYTHONPATH
    return harness


# ------------------------------------------------------------------
# Challenges
# ------------------------------------------------------------------

def _harness_sparse_parity(**kwargs):
    return _import_harness().measure_sparse_parity(**kwargs)


def _harness_sparse_sum(**kwargs):
    return _import_harness().measure_sparse_sum(**kwargs)


def _harness_sparse_and(**kwargs):
    return _import_harness().measure_sparse_and(**kwargs)


def register_default_challenges():
    """Register the three built-in challenges."""
    register_challenge(
        "sparse-parity",
        harness_fn=_harness_sparse_parity,
        description=(
            "Learn XOR/parity of k secret bits from {-1,+1}^n inputs. "
            "The 'drosophila' of energy-efficient training."
        ),
        default_config={
            "n_bits": 20, "k_sparse": 3, "hidden": 200,
            "lr": 0.1, "wd": 0.01, "batch_size": 32,
            "n_train": 1000, "max_epochs": 200, "seed": 42,
        },
    )

    register_challenge(
        "sparse-sum",
        harness_fn=_harness_sparse_sum,
        description=(
            "Learn sum of k secret bits. Output in [-k, k]. "
            "Each bit contributes independently (first-order structure)."
        ),
        default_config={
            "n_bits": 20, "k_sparse": 3, "hidden": 200,
            "lr": 0.1, "wd": 0.01, "batch_size": 32,
            "n_train": 1000, "max_epochs": 200, "seed": 42,
        },
    )

    register_challenge(
        "sparse-and",
        harness_fn=_harness_sparse_and,
        description=(
            "Learn AND of k secret bits. Maps {-1,+1} to {0,1} per bit, "
            "then takes product. Output is 1 only when ALL k secret bits are +1."
        ),
        default_config={
            "n_bits": 20, "k_sparse": 3, "hidden": 200,
            "lr": 0.1, "wd": 0.01, "batch_size": 32,
            "n_train": 1000, "max_epochs": 200, "seed": 42,
        },
    )


# ------------------------------------------------------------------
# Methods  (order must match the original METHOD_MAP indices for
#           backward compatibility with answer_key.json)
# ------------------------------------------------------------------

def register_default_methods():
    """Register the 16 built-in methods."""

    # Neural net approaches
    register_method(
        "sgd", category="neural_net",
        description="Standard backprop (fast.py)",
    )
    register_method(
        "perlayer", category="neural_net",
        description="Per-layer forward-backward",
    )
    register_method(
        "sign_sgd", category="neural_net",
        description="Sign of gradient, fixed step",
    )
    register_method(
        "curriculum", category="neural_net",
        description="Train small n first, expand",
    )
    register_method(
        "forward_forward", category="neural_net",
        description="Hinton's FF (known to fail at n=20)",
    )

    # Algebraic / exact
    register_method(
        "gf2", category="algebraic",
        description="Gaussian elimination over GF(2)",
    )
    register_method(
        "km", category="algebraic",
        description="Kushilevitz-Mansour influence estimation",
    )
    register_method(
        "smt", category="algebraic",
        description="Constraint solver / backtracking",
    )
    register_method(
        "fourier", category="algebraic",
        description="Walsh-Hadamard correlation",
    )

    # Information-theoretic
    register_method(
        "lasso", category="information_theoretic",
        description="L1 on interaction features",
    )
    register_method(
        "mdl", category="information_theoretic",
        description="Minimum description length",
    )
    register_method(
        "mutual_info", category="information_theoretic",
        description="Mutual information",
    )
    register_method(
        "random_proj", category="information_theoretic",
        description="Monte Carlo Fourier subsampling",
    )

    # Alternative framings
    register_method(
        "rl", category="alternative",
        description="Reinforcement learning bit querying",
    )
    register_method(
        "genetic_prog", category="alternative",
        description="Symbolic regression",
    )
    register_method(
        "evolutionary", category="alternative",
        description="Random/evolutionary subset search",
    )


def register_defaults():
    """Register all default challenges and methods."""
    register_default_challenges()
    register_default_methods()

"""
Registry for challenges and methods. Extensible without editing env.py.

To add a new challenge or method, call the register functions from your
own module or from default_registry.py. The environment looks up
challenges and methods through this registry at runtime.

Example:

    from sparse_parity.eval.registry import register_challenge, register_method

    register_challenge(
        "sparse-majority",
        harness_fn=my_measure_fn,
        description="Majority vote of k secret bits",
    )

    register_method(
        "voting_tree",
        category="tree",
        applicable_challenges=["sparse-majority"],
        description="Decision tree with majority vote splitting",
    )
"""

_challenges = {}
_methods = {}
# Stable ordering list -- appended in registration order.
# Used by the Gymnasium Discrete action space (index -> method name).
_method_order = []


def register_challenge(name, harness_fn, description="", default_config=None):
    """
    Register a challenge that the environment can use.

    Parameters
    ----------
    name : str
        Slug like "sparse-parity", "sparse-sum".
    harness_fn : callable
        Function with signature (method, n_bits, k_sparse, seed, **kw) -> dict.
        Must return at least: accuracy, ard, dmc, time_s, total_floats.
    description : str
        Human-readable one-liner.
    default_config : dict or None
        Default kwargs passed to harness_fn when none are provided.
    """
    _challenges[name] = {
        "harness_fn": harness_fn,
        "description": description,
        "default_config": default_config or {},
    }


def register_method(name, category="unknown", applicable_challenges=None,
                     description=""):
    """
    Register a method the agent can select.

    Parameters
    ----------
    name : str
        Slug like "sgd", "gf2", "km".
    category : str
        Grouping label (e.g. "neural_net", "algebraic", "information_theoretic").
    applicable_challenges : list[str] or None
        Which challenges this method works on. None means all.
    description : str
        Human-readable one-liner.
    """
    _methods[name] = {
        "category": category,
        "applicable_challenges": applicable_challenges,
        "description": description,
    }
    if name not in _method_order:
        _method_order.append(name)


def get_challenge(name):
    """Return challenge dict or raise KeyError."""
    if name not in _challenges:
        raise KeyError(
            f"Unknown challenge: {name!r}. "
            f"Registered challenges: {list(_challenges)}"
        )
    return _challenges[name]


def get_method(name):
    """Return method dict or raise KeyError."""
    if name not in _methods:
        raise KeyError(
            f"Unknown method: {name!r}. "
            f"Registered methods: {list(_methods)}"
        )
    return _methods[name]


def list_challenges():
    """Return list of registered challenge names (insertion order)."""
    return list(_challenges)


def list_methods():
    """Return list of registered method names in stable order."""
    return list(_method_order)


def get_method_index(name):
    """
    Return the stable integer index for a method name.

    This is the action the agent sends to env.step().
    Raises KeyError if the method is not registered.
    """
    try:
        return _method_order.index(name)
    except ValueError:
        raise KeyError(
            f"Unknown method: {name!r}. "
            f"Registered methods: {list(_method_order)}"
        )


def get_harness_fn(challenge_name):
    """Convenience: return the harness callable for a challenge."""
    return get_challenge(challenge_name)["harness_fn"]

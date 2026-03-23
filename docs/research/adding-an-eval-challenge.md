# How to Add a Challenge to the Eval Environment

Step-by-step guide for adding a new challenge to the Gymnasium eval
environment (`SutroYaro/SparseParity-v0`). This covers the registry,
harness, and answer key. An agent or contributor should be able to
complete this in one session.

For adding challenges to the *research workspace* (harness, docs,
search space), see [adding-a-challenge.md](adding-a-challenge.md).

## Overview

The eval environment uses a **registry** (`sparse_parity.eval.registry`)
so you never need to edit `env.py` to add a new challenge or method.

```
registry.py          -- register_challenge(), register_method()
default_registry.py  -- ships 3 challenges, 16 methods
env.py               -- reads the registry at runtime
backends.py          -- looks up harness functions from registry
```

## 1. Implement the harness measure function

Write a `measure_your_challenge(method, n_bits, k_sparse, seed, **kwargs)`
function that returns a dict with at least:

```python
{
    "accuracy": float,   # 0.0 to 1.0
    "ard": float or None,
    "dmc": float or None,
    "time_s": float,
    "total_floats": int or None,
    "found_secret": list or None,
}
```

You can put this function anywhere importable from `PYTHONPATH=src`.
The simplest option is adding it to `src/harness.py` (following the
existing `measure_sparse_sum` as a template), but you can also put it
in a separate module.

**Do not modify harness.py in experiment PRs** (LAB.md rule #9).
If you are adding infrastructure (not running an experiment), a
separate PR that modifies harness.py is fine.

## 2. Register the challenge

In your module, or in `default_registry.py` if this ships with the repo:

```python
from sparse_parity.eval.registry import register_challenge

def _my_harness(**kwargs):
    import my_module
    return my_module.measure_my_challenge(**kwargs)

register_challenge(
    "my-challenge",
    harness_fn=_my_harness,
    description="One-line description of the task",
    default_config={"n_bits": 20, "k_sparse": 3, "seed": 42},
)
```

The `harness_fn` is called by the `LocalBackend` with keyword arguments:
`method`, `n_bits`, `k_sparse`, `seed`, plus any extras.

## 3. Register methods for the challenge

```python
from sparse_parity.eval.registry import register_method

register_method(
    "my_method",
    category="algebraic",
    applicable_challenges=["my-challenge"],
    description="What this method does",
)
```

If a method works on all challenges, set `applicable_challenges=None`.

**Important**: Method registration order determines the action-space
index. The 16 default methods are indices 0-15. New methods get
indices 16, 17, etc. This means the action space grows automatically.

## 4. Add answer key entries

Add experiments to `src/sparse_parity/eval/answer_key.json` so the
`OracleAgent` and `DiscoveryGrader` know the ground truth:

```json
{
    "exp_id": "my-exp1",
    "method": "my_method",
    "challenge": "my-challenge",
    "accuracy": 1.0,
    "ard": 500.0,
    "dmc": 1200.0,
    "category": "algebraic",
    "result": "SOLVED"
}
```

## 5. Run baselines

```bash
PYTHONPATH=src python3 -c "
from sparse_parity.eval.registry import register_challenge, register_method
# ... your registrations ...

import gymnasium as gym
import sparse_parity.eval

env = gym.make('SutroYaro/SparseParity-v0',
    challenge='my-challenge', metric='dmc', budget=5)
obs, info = env.reset()
print(info)
obs, r, _, _, info = env.step(0)
print(info)
"
```

## 6. External registration (no repo changes needed)

If you are developing outside the repo, you can register challenges
and methods at runtime before creating the environment:

```python
import sparse_parity.eval  # loads defaults
from sparse_parity.eval import registry

registry.register_challenge("my-challenge", harness_fn=my_fn)
registry.register_method("my-method", category="custom")

env = gym.make("SutroYaro/SparseParity-v0",
    challenge="my-challenge", metric="dmc", budget=10)
```

## Checklist

- [ ] Harness measure function implemented and tested standalone
- [ ] Challenge registered via `register_challenge()`
- [ ] At least 1 method registered via `register_method()`
- [ ] Answer key entries added (if shipping with the repo)
- [ ] Baselines recorded in DISCOVERIES.md
- [ ] Environment creates and runs without errors
- [ ] Existing tests (`run_eval.py`) still pass

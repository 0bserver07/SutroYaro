# Gymnasium Environment Interface Specification

> SutroYaro/SparseParity-v0 -- Can an AI agent do energy-efficient ML research?

## Motivation

Standard RL environments test game-playing or robot control. This environment
tests something different: whether an agent can navigate a *research search
space* to find energy-efficient solutions to a learning problem.

The agent's "game" is method selection. Given a learning challenge (sparse
parity, sparse sum, sparse AND), the agent picks methods and observes energy
metrics. The goal is to find the lowest-cost solution within a fixed budget
of experiments. This directly mirrors how a human researcher explores the
method landscape documented in DISCOVERIES.md.

### How this differs from typical RL environments

| Property | Typical RL (Atari, MuJoCo) | SutroYaro |
|----------|---------------------------|-----------|
| Episode length | 100s-1000s of steps | 5-30 steps (budget) |
| Action semantics | Joystick/torque | "Try method X" |
| State transitions | Physics/game rules | Stochastic (method success depends on problem structure) |
| Reward signal | Score/distance | Improvement in energy metric (ARD or DMC) |
| Optimal policy | Reflexive control | Sequential experiment design |
| Ground truth | Unknown | Known (34 experiments, DISCOVERIES.md) |
| Reset cost | Cheap | Moderate (each step runs an actual experiment, ~0.001-0.5s) |

The key insight: an episode is a *research trajectory*, not a game trajectory.
The agent must decide which experiments to run given what it has learned so far.
A good agent avoids redundant experiments and converges on the best method early.

---

## Variant A: Method Selection (Discrete Actions)

### Quick Start

```python
import gymnasium as gym

env = gym.make("SutroYaro/SparseParity-v0",
    challenge="sparse-parity",
    n_bits=20,
    k_sparse=3,
    metric="dmc",
    budget=20,
    seed=42,
)

obs, info = env.reset()
# Agent picks a method index
action = 5  # e.g., index for "gf2"
obs, reward, terminated, truncated, info = env.step(action)

# info contains the raw harness output
print(info)
# {"method": "gf2", "accuracy": 1.0, "ard": 420.0, "dmc": 8607.0,
#  "time_s": 0.000509, "total_floats": 860}
```

### Registration

```python
gymnasium.register(
    id="SutroYaro/SparseParity-v0",
    entry_point="sparse_parity.eval.env:SutroYaroEnv",
)
```

---

### Observation Space

```python
gymnasium.spaces.Dict({
    # Problem description (fixed for the episode)
    "challenge": gymnasium.spaces.Discrete(3),
    #   0 = sparse-parity, 1 = sparse-sum, 2 = sparse-and

    "n_bits": gymnasium.spaces.Discrete(101, start=3),
    #   Range: 3..100 (matches search_space.yaml values)

    "k_sparse": gymnasium.spaces.Discrete(11, start=3),
    #   Range: 3..10

    # Optimization target (fixed for the episode)
    "metric": gymnasium.spaces.Discrete(2),
    #   0 = ard, 1 = dmc

    # Dynamic state (changes every step)
    "best_score": gymnasium.spaces.Box(
        low=0.0, high=1e12, shape=(1,), dtype=np.float32
    ),
    #   Best value of the target metric seen so far.
    #   Initialized to float("inf"). Lower is better.
    #   Only counts experiments where accuracy >= 0.95.

    "budget_remaining": gymnasium.spaces.Discrete(101),
    #   Steps left before truncation.

    "steps_taken": gymnasium.spaces.Discrete(101),
    #   How many experiments have been run.

    "methods_tried": gymnasium.spaces.MultiBinary(16),
    #   Boolean mask: 1 if method index i has been tried.
    #   Length = number of methods (16).

    "last_result": gymnasium.spaces.Dict({
        "method_index": gymnasium.spaces.Discrete(16),
        "accuracy":     gymnasium.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        "ard":          gymnasium.spaces.Box(0.0, 1e12, shape=(1,), dtype=np.float32),
        "dmc":          gymnasium.spaces.Box(0.0, 1e12, shape=(1,), dtype=np.float32),
        "time_s":       gymnasium.spaces.Box(0.0, 1e6, shape=(1,), dtype=np.float32),
        "solved":       gymnasium.spaces.Discrete(2),  # 0=no, 1=yes (acc >= 0.95)
    }),
    #   Result of the most recent experiment.
    #   On reset, all values are 0.
})
```

### Action Space

```python
gymnasium.spaces.Discrete(16)
```

Each integer maps to a method from `search_space.yaml`:

| Index | Method | Category | Notes |
|-------|--------|----------|-------|
| 0 | `sgd` | Neural net | Standard backprop, ~0.12s |
| 1 | `perlayer` | Neural net | Per-layer forward-backward |
| 2 | `sign_sgd` | Neural net | Sign of gradient, fixed step |
| 3 | `curriculum` | Neural net | Train small n first, expand |
| 4 | `forward_forward` | Neural net | Hinton's FF (fails at n=20) |
| 5 | `gf2` | Algebraic | Gaussian elimination over GF(2) |
| 6 | `km` | Algebraic | Kushilevitz-Mansour influence |
| 7 | `smt` | Algebraic | Constraint solver / backtracking |
| 8 | `fourier` | Algebraic | Walsh-Hadamard correlation |
| 9 | `lasso` | Info-theoretic | L1 on interaction features |
| 10 | `mdl` | Info-theoretic | Minimum description length |
| 11 | `mutual_info` | Info-theoretic | Mutual information |
| 12 | `random_proj` | Info-theoretic | Monte Carlo Fourier subsampling |
| 13 | `rl` | Alternative | RL bit querying |
| 14 | `genetic_prog` | Alternative | Symbolic regression |
| 15 | `evolutionary` | Alternative | Random/evolutionary subset search |

This mapping is derived from `research/search_space.yaml` and is fixed for v0.

### Reward Function

The reward encourages finding better solutions and penalizes failures.

```python
def compute_reward(self, result, previous_best):
    accuracy = result.get("accuracy", 0.0)
    score = result.get(self.metric)  # ard or dmc

    if accuracy < 0.95:
        # Method did not solve the problem
        return -0.1

    if score is None:
        # Method solved but metric not available (shouldn't happen)
        return 0.0

    if previous_best == float("inf"):
        # First successful solve -- large positive reward
        # Normalize: log scale because metrics span 6 orders of magnitude
        return 10.0 / (1.0 + math.log10(max(score, 1.0)))

    if score < previous_best:
        # Improvement -- reward proportional to relative improvement
        improvement_ratio = (previous_best - score) / previous_best
        return 10.0 * improvement_ratio

    # No improvement over previous best
    return -0.01
```

Reward design rationale:

- **Log-scale first solve**: Metrics range from 20 (KM-min ARD) to 78 billion
  (Fourier DMC). Raw differences would make early rewards dominate. The log
  normalization keeps first-solve rewards in a comparable range regardless of
  which method the agent tries first.
- **Relative improvement**: After the first solve, reward is proportional to
  the fraction of improvement. Going from DMC 20,633 to 3,578 (KM to KM-min)
  yields reward ~8.3. Going from 8,607 to 3,578 yields ~5.8.
- **Small penalty for failure**: -0.1 per failed method. Enough to discourage
  random flailing, small enough that exploration is not crushed.
- **Tiny penalty for no improvement**: -0.01 for trying a method that works
  but does not improve the best score. Discourages repeating what is already
  known without strongly penalizing exploration.

### Episode Termination

```python
terminated = False  # Never terminated early by "winning"
truncated = (self.steps_taken >= self.budget)
```

- **No early termination**: The agent is always allowed to use its full budget.
  Even after finding the optimal method, it might discover something better by
  trying a different approach or parameter setting. The environment does not
  know the true optimum.
- **Truncation at budget**: When `budget_remaining` hits 0, the episode ends
  with `truncated=True`. Default budget is 20 steps.

### Info Dict

Every `step()` returns an `info` dict with the raw harness output:

```python
info = {
    # From harness.py
    "method": "km",           # string name
    "accuracy": 1.0,          # float, 0.0-1.0
    "ard": 92.0,              # float, weighted average reuse distance
    "dmc": 20633.0,           # float, data movement complexity
    "time_s": 0.006,          # float, wall-clock seconds
    "total_floats": 4420,     # int, total float accesses
    "challenge": "sparse-parity",

    # Added by environment
    "found_secret": [3, 7, 15],  # list of ints (if method exposes it)
    "is_new_best": True,         # bool, was this the new best score?
    "improvement": 0.83,         # float, relative improvement (0 if no improvement)
    "error": None,               # string if method failed/crashed
}
```

On `reset()`:

```python
info = {
    "challenge": "sparse-parity",
    "n_bits": 20,
    "k_sparse": 3,
    "metric": "dmc",
    "budget": 20,
    "methods": [
        "sgd", "perlayer", "sign_sgd", "curriculum", "forward_forward",
        "gf2", "km", "smt", "fourier", "lasso", "mdl", "mutual_info",
        "random_proj", "rl", "genetic_prog", "evolutionary"
    ],
}
```

---

### Constructor Parameters

```python
env = gym.make("SutroYaro/SparseParity-v0",
    challenge="sparse-parity",  # "sparse-parity" | "sparse-sum" | "sparse-and"
    n_bits=20,                  # int, from search_space.yaml values
    k_sparse=3,                 # int, from search_space.yaml values
    metric="dmc",               # "ard" | "dmc" -- what the agent optimizes
    budget=20,                  # int, max steps per episode
    seed=42,                    # int, controls secret bits and data generation
    harness_timeout=10.0,       # float, max seconds per method call
)
```

### Internal Mechanics

On each `step(action)`:

1. Map action index to method name via the fixed table.
2. Call `harness.measure_sparse_parity(method, n_bits, k_sparse, seed=self.seed)`
   (or `measure_sparse_sum`/`measure_sparse_and` depending on challenge).
3. If the method call raises an exception or times out, return accuracy=0,
   ard=None, dmc=None with the error in `info["error"]`.
4. Update `best_score` if accuracy >= 0.95 and the metric improved.
5. Update `methods_tried` mask.
6. Compute reward.
7. Advance step counter.

---

### Ground Truth and Evaluation

The optimal policy for this environment is known from DISCOVERIES.md.
For `metric="dmc"`, `challenge="sparse-parity"`, `n_bits=20`, `k_sparse=3`:

| Rank | Method | DMC | Steps to discover |
|------|--------|-----|-------------------|
| 1 | KM-min (1 sample) | 3,578 | Optimal: 1 step |
| 2 | KM (5 samples) | 20,633 | If agent tries KM |
| 3 | GF2 | 8,607 | If agent tries GF2 |
| 4 | SMT | 348,336 | -- |
| 5 | SGD | 1,278,460 | -- |
| 6 | Fourier | 78,140,662,852 | -- |

An oracle agent that already knows the answer picks action 6 (km) on step 1
and is done. A random agent would take ~16 steps on average to try all methods.
The interesting question is whether an RL agent can learn to identify good
methods in fewer steps than random.

### Multi-challenge evaluation

To test generalization, run episodes across all three challenges:

```python
for challenge in ["sparse-parity", "sparse-sum", "sparse-and"]:
    env = gym.make("SutroYaro/SparseParity-v0",
        challenge=challenge, metric="dmc", budget=20, seed=42)
    obs, info = env.reset()
    # ... agent plays ...
```

The optimal method differs per challenge:
- **sparse-parity**: KM-min (DMC 3,578)
- **sparse-sum**: SGD (DMC 2,862)
- **sparse-and**: KM with 20 samples (DMC TBD, but KM is best ARD)

A good agent learns that method rankings change across challenges.

---

## Variant B: Code Generation (Stretch Goal)

> Status: Design only. Not planned for implementation in the near term.

In Variant B, the agent writes Python code as its action. The environment
executes the code in a sandbox and measures the result using the same harness
metrics.

### Interface Sketch

```python
env = gym.make("SutroYaro/SparseParityCode-v0",
    challenge="sparse-parity",
    n_bits=20, k_sparse=3,
    metric="dmc",
    budget=10,      # fewer steps because each is expensive
    seed=42,
    sandbox="subprocess",   # "subprocess" | "docker"
    max_code_length=2000,   # characters
    exec_timeout=30.0,      # seconds per code execution
)

obs, info = env.reset()
# obs includes a text prompt describing the challenge and available imports

action = """
import numpy as np
from sparse_parity.config import Config
from sparse_parity.tracker import MemTracker

def solve(n_bits, k_sparse, seed):
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    # ... agent's custom solution ...
    return {"accuracy": acc, "found_secret": found}
"""

obs, reward, terminated, truncated, info = env.step(action)
```

### Action Space

```python
gymnasium.spaces.Text(
    min_length=10,
    max_length=2000,
)
```

The agent submits a Python string that defines a `solve(n_bits, k_sparse, seed)`
function. The environment:

1. Writes the code to a temp file.
2. Executes it in a sandboxed subprocess with a timeout.
3. Wraps it with MemTracker to measure ARD/DMC.
4. Returns the same observation/reward/info structure as Variant A.

### Observation Space Changes

Same as Variant A, plus:

```python
"last_code": gymnasium.spaces.Text(max_length=2000),
#   The code the agent submitted last step (for self-correction).

"last_error_msg": gymnasium.spaces.Text(max_length=500),
#   Traceback if the code crashed. Empty string if successful.
```

### Why This is Hard

- **Security**: Executing arbitrary code requires sandboxing. Even with
  subprocess isolation, the agent could attempt file I/O, network access,
  or resource exhaustion.
- **MemTracker integration**: The agent's code must be instrumented for
  ARD/DMC measurement. Either the agent includes tracker calls (unlikely
  for a learned policy), or the environment auto-instruments the code
  (complex and fragile).
- **Action space size**: The space of valid Python programs is enormous.
  Standard RL algorithms (PPO, DQN) cannot handle text actions. This
  variant is designed for LLM-based agents, not traditional RL.
- **Reward attribution**: If the code crashes, was it a syntax error, a logic
  error, or a timeout? The reward function needs to distinguish these.

### When to Pursue Variant B

Variant B becomes worthwhile when:
1. Variant A agents plateau (they learn the optimal method but cannot improve
   the method itself).
2. LLM-based agents with code generation (e.g., Voyager-style) are available
   as Gymnasium-compatible policies.
3. The sandbox infrastructure is battle-tested on simpler code-generation
   environments first.

---

## File Layout

```
src/sparse_parity/eval/
    README.md       <-- this file (interface spec)
    __init__.py     <-- gymnasium registration (issue #34)
    env.py          <-- SutroYaroEnv implementation (issue #34)
```

## Dependencies

```
gymnasium >= 1.0.0
numpy >= 1.24.0
```

The environment imports from the existing codebase:
- `src/harness.py` (locked, not modified)
- `src/sparse_parity/config.py`
- `src/sparse_parity/tracker.py`

## References

- Gymnasium API: https://gymnasium.farama.org/api/env/
- search_space.yaml: `research/search_space.yaml`
- Harness: `src/harness.py`
- Ground truth: `DISCOVERIES.md`
- GitHub Issue: #33 (this spec), #34 (implementation)

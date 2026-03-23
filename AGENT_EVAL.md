# Eval Environment Guide

Machine-readable guide for coding agents (Claude Code, Codex, Gemini CLI).

## Quick test

```bash
PYTHONPATH=src python3 -c "
import gymnasium as gym
import sparse_parity.eval
env = gym.make('SutroYaro/SparseParity-v0', metric='dmc', budget=10)
obs, info = env.reset()
print('Methods:', info['methods'])
obs, r, _, _, info = env.step(5)  # try gf2
print(f'{info[\"method\"]}: DMC={info[\"dmc\"]}, reward={r:.2f}')
env.render()
"
```

Prerequisites: `pip install gymnasium numpy`

## What this environment tests

An AI agent picks methods to solve a learning problem and observes energy metrics. The goal is to find the lowest-cost method within a fixed experiment budget.

- 3 challenges: `sparse-parity`, `sparse-sum`, `sparse-and`
- 16 methods (all runnable: 5 via harness, 9 via live fallbacks, 2 cached)
- Metrics: ARD (average reuse distance) and DMC (data movement complexity)
- 36 experiments as ground truth, 72-point discovery grading (12 categories)
- Episode = research trajectory (5-30 steps), not a game

## Action space

`Discrete(16)` -- each integer maps to a method:

| Index | Method | Category | Implemented |
|-------|--------|----------|-------------|
| Index | Method | Category | Source |
|-------|--------|----------|--------|
| 0 | `sgd` | neural_net | harness |
| 1 | `perlayer` | neural_net | live fallback |
| 2 | `sign_sgd` | neural_net | live fallback |
| 3 | `curriculum` | neural_net | live fallback |
| 4 | `forward_forward` | neural_net | cached (fails at 58.5%, too slow live) |
| 5 | `gf2` | algebraic | harness |
| 6 | `km` | algebraic | harness |
| 7 | `smt` | algebraic | harness |
| 8 | `fourier` | algebraic | harness |
| 9 | `lasso` | information_theoretic | live fallback |
| 10 | `mdl` | information_theoretic | live fallback |
| 11 | `mutual_info` | information_theoretic | live fallback |
| 12 | `random_proj` | information_theoretic | live fallback |
| 13 | `rl` | alternative | cached (Q-learning too slow) |
| 14 | `genetic_prog` | alternative | live fallback |
| 15 | `evolutionary` | alternative | live fallback |

## Constructor parameters

```python
env = gym.make("SutroYaro/SparseParity-v0",
    challenge="sparse-parity",  # "sparse-parity" | "sparse-sum" | "sparse-and"
    n_bits=20,                  # int, 3..100
    k_sparse=3,                 # int, 3..10
    metric="dmc",               # "ard" | "dmc"
    budget=20,                  # int, max steps per episode
    seed=42,                    # int
    harness_timeout=10.0,       # float, max seconds per method call
)
```

Multi-challenge variant:

```python
env = gym.make("SutroYaro/MultiChallenge-v0",
    budget_per=10,              # budget per challenge
    n_bits=20, k_sparse=3, metric="dmc",
)
```

## Running the full evaluation

```bash
PYTHONPATH=src python3 src/sparse_parity/eval/run_eval.py
```

Runs 3 baseline agents (Random, Greedy, Oracle) x 5 episodes in ~20 seconds (all 16 methods run). Outputs to `results/eval/baselines.json` and `results/eval/multi_challenge.json`.

## How to add a new method

1. Register it before creating the environment:

```python
from sparse_parity.eval.registry import register_method

register_method(
    "my_method",
    category="algebraic",           # "neural_net" | "algebraic" | "information_theoretic" | "alternative"
    applicable_challenges=["sparse-parity"],  # None = all challenges
    description="What this method does",
)
```

2. Method registration order determines action index. Default methods are 0-15. New methods get 16, 17, etc.

3. Add answer key entry to `src/sparse_parity/eval/answer_key.json`:

```json
{
    "exp_id": "my-exp1",
    "method": "my_method",
    "challenge": "sparse-parity",
    "accuracy": 1.0,
    "ard": 500.0,
    "dmc": 1200.0,
    "category": "algebraic",
    "result": "SOLVED"
}
```

4. Re-run baselines: `PYTHONPATH=src python3 src/sparse_parity/eval/run_eval.py`

## How to add a new challenge

See `docs/research/adding-an-eval-challenge.md` for the full guide. Summary:

1. Write `measure_my_challenge(method, n_bits, k_sparse, seed, **kwargs) -> dict`
2. Register: `register_challenge("my-challenge", harness_fn=my_fn, description="...")`
3. Register methods for it
4. Add answer key entries
5. Run baselines

## Compute backends

| Backend | How to use | Notes |
|---------|-----------|-------|
| Local (default) | `gym.make(...)` | Direct Python import, runs harness locally |
| Modal | Prototype only | Requires `pip install modal`, MODAL_TOKEN_ID/SECRET env vars |
| Remote | Prototype only | HTTP POST to hosted harness endpoint |

Backend selection is handled by `sparse_parity.eval.backends.get_backend(name)`.

## Discovery grading categories

The `DiscoveryGrader` scores research quality across 12 categories, 72 points total. Each category measures a specific research discovery.

| Category | Pts | How to earn |
|----------|-----|-------------|
| Discovered algebraic solver | 10 | Try GF2, KM, or SMT and solve. Partial credit (3) for trying without solving. |
| Discovered KM influence | 7 | Solve with KM. This is the O(n) approach vs O(C(n,k)) for Fourier. |
| Identified local learning failure | 5 | Try forward_forward and observe it fails (acc < 95%). |
| Found metric disagreement | 5 | Solve with both KM (ARD best) and GF2 (DMC best). |
| Found curriculum speedup | 5 | Solve with curriculum learning. |
| Identified parity invisibility | 5 | Observe 2+ method failures and also find working methods. The contrast reveals parity structure. |
| Exploration breadth | 5 | 1 pt per distinct method that solves (acc >= 95%), max 5. |
| Efficiency | 5 | 5 pts if best method found in steps 1-3. Decreasing to 0 at step 16. |
| Optimized beyond baseline | 3 | Find any method with DMC below SGD baseline (1,278,460). |
| Cross-challenge analysis | 3 | MultiChallengeEnv only. Solve methods across 2+ challenges. |
| Cache model insight | 3 | Measure DMC across 3+ methods with different values. |
| Correct failure classification | 2/each | Per failed method: 1 pt for observing, 2 pts if agent tried alternatives after. Max 16. |

**Total: 72 points.**

Usage:

```python
from sparse_parity.eval.grader import DiscoveryGrader

grader = DiscoveryGrader()
report = grader.grade(env.experiment_log, challenge="sparse-parity")
print(report)
print(f"Score: {report.total_score}/{report.max_possible} ({report.percentage:.0f}%)")
```

## Reward function

```
accuracy < 0.95         -> -0.1  (method failed)
first successful solve  -> 10 / (1 + log10(max(score, 1)))
improved best score     -> 10 * (previous_best - score) / previous_best
no improvement          -> -0.01
```

## Ground truth (sparse-parity, n=20, k=3)

| Method | DMC | ARD | Time |
|--------|-----|-----|------|
| KM-min (1 sample) | 3,578 | 20 | ~0.001s |
| GF2 | 8,607 | ~420 | 509 us |
| KM (5 samples) | 20,633 | 92 | 0.001-0.006s |
| SMT | 348,336 | 3,360 | 0.002s |
| SGD | 1,278,460 | 8,504 | 0.12s |
| Fourier | 78,140,662,852 | -- | -- |

## Files

| File | Purpose |
|------|---------|
| `src/sparse_parity/eval/__init__.py` | Gymnasium registration, imports defaults |
| `src/sparse_parity/eval/env.py` | `SutroYaroEnv` and `MultiChallengeEnv` implementation |
| `src/sparse_parity/eval/registry.py` | `register_challenge()`, `register_method()` |
| `src/sparse_parity/eval/default_registry.py` | Ships 3 challenges, 16 methods |
| `src/sparse_parity/eval/backends.py` | Local, Modal, Remote compute backends |
| `src/sparse_parity/eval/baselines.py` | Random, Greedy, Oracle agents |
| `src/sparse_parity/eval/grader.py` | Discovery scoring (72-point rubric, 12 categories) |
| `src/sparse_parity/eval/answer_key.json` | Ground truth: 36 experiments, 12 negative results |
| `src/sparse_parity/eval/run_eval.py` | Evaluation script (3 agents x 5 episodes) |
| `src/sparse_parity/eval/README.md` | Full interface specification |
| `docs/research/eval-environment.md` | Human-readable docs page |
| `docs/research/adding-an-eval-challenge.md` | Guide for adding challenges |

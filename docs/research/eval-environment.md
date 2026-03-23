# Research Eval Environment

A Gymnasium environment for testing whether a coding agent can figure out how to solve a learning problem efficiently. The agent picks methods, runs real experiments, and gets scored on what it discovered.

## Quick start

```bash
git clone https://github.com/cybertronai/SutroYaro
cd SutroYaro
pip install gymnasium numpy
PYTHONPATH=src python3 -c "
import gymnasium as gym
import sparse_parity.eval.env

env = gym.make('SutroYaro/SparseParity-v0', metric='dmc', budget=16)
obs, info = env.reset()
obs, reward, done, trunc, info = env.step(5)  # try GF(2)
print(f'Method: {info[\"method\"]}, DMC: {info[\"dmc\"]}, Reward: {reward:.2f}')
env.render()
"
```

Or point your coding agent at the repo and ask it to run the eval:

```bash
claude --dangerously-skip-permissions
# then: "run the eval environment and tell me what you find"
```

## What this tests

The agent picks methods to solve sparse parity (and two related problems). Each step calls real experiment code and returns energy metrics. The agent has a fixed budget of experiments and needs to find the cheapest method.

The problem is small (solves in under a second) but the method space is large (16 options across 4 categories) and many methods fail. An agent that tries SGD first, then discovers GF(2) solves it 240x faster, then notices KM has better reuse distance but worse total data movement, has made three real research discoveries.

We know the answers. 34 experiments have been run. The grading rubric checks whether the agent rediscovered what we already know.

## How this compares to existing benchmarks

Most agent benchmarks test code generation. PrimeIntellect's 105 community environments (scicode, gpu_puzzles, llm_training_puzzles) ask agents to write code and check if it runs. ScienceAgentBench (Allen AI, 102 tasks) asks agents to reproduce paper results. HuggingFace has model leaderboards but no agent research environments.

This environment tests something different: experiment selection. The agent does not write code. It picks which method to run, observes the result, and decides what to try next. The grading rubric checks whether the agent made specific discoveries (found the algebraic solver, noticed the metric disagreement, observed that local learning fails), not whether it produced correct code.

We can grade this because we have ground truth. Most research environments cannot score an agent's trajectory because the optimal policy is unknown. Here, 34 experiments establish what works, what fails, and why.

## Environments

| Registration | What it does |
|-------------|-------------|
| `SutroYaro/SparseParity-v0` | One challenge at a time (parity, sum, or AND) |
| `SutroYaro/MultiChallenge-v0` | Cycles through all three per episode |

## Action space: 16 methods

| Index | Method | Category | Source | Solves parity? |
|-------|--------|----------|--------|---------------|
| 0 | SGD | Neural net | harness | Yes (0.12s) |
| 1 | Per-layer | Neural net | live fallback | Slow (often times out) |
| 2 | Sign SGD | Neural net | live fallback | Slow (often times out) |
| 3 | Curriculum | Neural net | live fallback | Yes |
| 4 | Forward-Forward | Neural net | cached | No (58.5% max) |
| 5 | GF(2) | Algebraic | harness | Yes (509us) |
| 6 | KM Influence | Algebraic | harness | Yes |
| 7 | SMT | Algebraic | harness | Yes |
| 8 | Fourier | Algebraic | harness | Yes |
| 9 | LASSO | Info-theoretic | live fallback | Yes |
| 10 | MDL | Info-theoretic | live fallback | Yes |
| 11 | Mutual Info | Info-theoretic | live fallback | Yes |
| 12 | Random Proj | Info-theoretic | live fallback | Yes |
| 13 | RL | Alternative | cached | Yes (cached) |
| 14 | Genetic Prog | Alternative | live fallback | Usually no |
| 15 | Evolutionary | Alternative | live fallback | Yes |

"Source" means how the method runs. Harness methods go through the locked evaluation harness. Live fallback methods run their own implementation. Cached methods return documented results because they're too slow for live eval (forward_forward needs 30+ seconds, RL Q-learning needs 50K episodes).

Methods that fail are part of the environment. An agent that tries forward_forward, observes 58.5% accuracy, and moves on has learned something about the problem structure.

## Baselines

| Agent | Mean Reward | Discovery Score | What it does |
|-------|------------|-----------------|-------------|
| Random | 16.61 | 49.4/72 (68.6%) | Picks random methods each step |
| Greedy | 16.91 | 57.0/72 (79.2%) | Tries each method in order, repeats the best |
| Oracle | 7.59 | 57.4/72 (79.7%) | Picks the best method first (from answer key) |

Oracle gets the lowest reward but highest discovery score. The reward function gives points for improvement (going from SGD to GF2), so finding the best method first leaves no room to improve. The discovery grader does not care about order.

## Discovery grading (12 categories, 72 points)

The grader checks what the agent figured out, not just what number it got.

| Category | Pts | What the agent needs to do |
|----------|-----|---------------------------|
| Discovered algebraic solver | 10 | Try GF2, KM, or SMT. Solve with one of them. (3 pts partial credit for trying without solving.) |
| Discovered KM influence | 7 | Solve with KM. This is the O(n) method. |
| Identified local learning failure | 5 | Try forward_forward. Observe accuracy < 95%. |
| Found metric disagreement | 5 | Solve with both KM and GF2. KM wins on ARD, GF2 wins on DMC. Both in the log means the agent has the data to notice. |
| Found curriculum speedup | 5 | Solve with curriculum. |
| Identified parity invisibility | 5 | Observe 2+ failures and also find working methods. The contrast reveals the problem structure. |
| Exploration breadth | 5 | 1 pt per method that solves the problem. Max 5. |
| Efficiency | 5 | Find the best method early. 5 pts in steps 1-3, decreasing to 0 at step 16. |
| Optimized beyond baseline | 3 | Find any method with DMC below SGD (1,278,460). |
| Cross-challenge analysis | 3 | MultiChallengeEnv only. Solve across 2+ challenges. |
| Cache model insight | 3 | Measure DMC across 3+ methods. The spread reveals cache/energy behavior. |
| Correct failure classification | 2/each | Per failed method: 1 pt for observing, 2 pts for moving on to try alternatives. Max 16. |

## Adding methods and challenges

The environment uses a registry. You can add methods or challenges without editing env.py.

```python
from sparse_parity.eval.registry import register_method, register_challenge

register_method("my_method", category="algebraic",
    applicable_challenges=["sparse-parity"])

register_challenge("my-challenge", harness_fn=my_fn,
    description="What it tests")
```

New methods get the next action index (16, 17, ...). See `docs/research/adding-an-eval-challenge.md` for the full walkthrough.

## Compute backends

The environment runs experiments locally by default. Two other backends exist as prototypes.

| Backend | How to use | Status |
|---------|-----------|--------|
| Local | `gym.make(...)` | Working. All 16 methods run. |
| Modal | `gym.make(..., backend="modal")` | Prototype. Returns error if Modal not configured. For GPU methods at larger scale. |
| Remote | `gym.make(..., backend="http://...")` | Prototype. HTTP POST to a hosted endpoint. For leaderboards. |

## Platform adapters

Four adapters exist for running the environment through different systems.

**Anthropic tool-use.** LLM agents interact via tool calls (run_experiment, check_status, read_experiment_log) instead of discrete indices. Includes a system prompt and grading.

```python
from sparse_parity.eval.adapters.anthropic_tools import AnthropicToolAdapter
adapter = AnthropicToolAdapter(challenge="sparse-parity", metric="dmc", budget=20)
tools = adapter.get_tools()
result = adapter.handle_tool_call("run_experiment", {"method": "gf2"})
grade = adapter.grade()
```

**PrimeIntellect verifiers.** Wraps the environment as a `vf.SingleTurnEnv` for their Environments Hub. Standalone test works without verifiers installed.

```bash
PYTHONPATH=src python3 src/sparse_parity/eval/adapters/primeintellect.py
```

**HuggingFace Spaces.** Gradio app with leaderboard table, interactive method selection, grading breakdown, and answer key viewer.

```bash
pip install gradio
PYTHONPATH=src python3 src/sparse_parity/eval/adapters/huggingface.py
```

**UK AISI Inspect.** Prototype task definition for the Inspect evaluation framework.

## Running the full evaluation

```bash
PYTHONPATH=src python3 src/sparse_parity/eval/run_eval.py
```

Runs 3 agents x 5 episodes in ~20 seconds. Outputs to `results/eval/baselines.json` and `results/eval/multi_challenge.json`.

## Answer key

`src/sparse_parity/eval/answer_key.json` has 36 experiments, 12 negative results, and the grading rubric. The experiments come from DISCOVERIES.md. The negative results explain why specific methods fail (Hebbian can't detect k-th order interactions, Forward-Forward's greedy layer-wise learning can't coordinate multi-layer feature extraction, etc.).

## Files

| File | What it is |
|------|-----------|
| `eval/env.py` | SparseParity-v0 and MultiChallenge-v0 |
| `eval/baselines.py` | Random, Greedy, Oracle agents |
| `eval/grader.py` | 12 categories, 72 points |
| `eval/answer_key.json` | 36 experiments, 12 negative results |
| `eval/registry.py` | Add challenges/methods at runtime |
| `eval/default_registry.py` | Ships 3 challenges, 16 methods |
| `eval/backends.py` | Local + 11 fallback method implementations, Modal and Remote prototypes |
| `eval/run_eval.py` | Evaluation script |
| `eval/README.md` | Full interface spec (observation space, action space, reward function) |
| `eval/adapters/anthropic_tools.py` | Claude tool-use adapter |
| `eval/adapters/primeintellect.py` | PrimeIntellect verifiers adapter |
| `eval/adapters/huggingface.py` | Gradio leaderboard app |
| `eval/adapters/inspect_task.py` | UK AISI Inspect prototype |
| `AGENT_EVAL.md` | Guide for coding agents |

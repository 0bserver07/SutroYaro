# Research Eval Environment

Can an AI agent do energy-efficient ML research? This Gymnasium-compatible environment tests that.

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

# Try GF(2) Gaussian elimination
obs, reward, done, trunc, info = env.step(5)
print(f'Method: {info[\"method\"]}, DMC: {info[\"dmc\"]}, Reward: {reward:.2f}')
env.render()
"
```

## What the agent does

Each step, the agent picks a method (discrete action, 16 options). The environment runs the method on sparse parity using the real evaluation harness and returns the energy metric (ARD or DMC). The agent's goal: find the lowest-cost method within a fixed budget of experiments.

This is a research trajectory, not a game. 5-30 steps per episode, not hundreds. The optimal policy is sequential experiment design.

## Two environments

| Environment | Registration | What it tests |
|-------------|-------------|---------------|
| `SutroYaro/SparseParity-v0` | Single challenge | Find best method for one problem |
| `SutroYaro/MultiChallenge-v0` | All three challenges | Generalize across parity, sum, AND |

## Methods (action space)

| Index | Method | Category |
|-------|--------|----------|
| 0 | SGD | Neural net |
| 1 | Per-layer | Neural net |
| 2 | Sign SGD | Neural net |
| 3 | Curriculum | Neural net |
| 4 | Forward-Forward | Neural net |
| 5 | GF(2) | Algebraic |
| 6 | KM Influence | Algebraic |
| 7 | SMT | Algebraic |
| 8 | Fourier | Algebraic |
| 9 | LASSO | Info-theoretic |
| 10 | MDL | Info-theoretic |
| 11 | Mutual Info | Info-theoretic |
| 12 | Random Proj | Info-theoretic |
| 13 | RL | Alternative |
| 14 | Genetic Prog | Alternative |
| 15 | Evolutionary | Alternative |

All 16 methods are runnable. 5 go through the locked harness (SGD, GF2, KM, SMT, Fourier). 9 run live via fallback implementations (perlayer, sign_sgd, curriculum, lasso, mdl, mutual_info, random_proj, genetic_prog, evolutionary). 2 return cached results because they're too slow for live eval (forward_forward, rl).

Methods that fail on sparse parity (forward_forward, genetic_prog, perlayer, sign_sgd) return low accuracy. This is correct behavior and itself a research signal -- the agent should observe these failures and learn from them.

## Baseline results

| Agent | Mean Reward | Discovery Score | Best Method |
|-------|------------|-----------------|-------------|
| Oracle | 7.59 | 57.4/72 (79.7%) | GF2 |
| Greedy | 16.91 | 57.0/72 (79.2%) | GF2 |
| Random | 16.61 | 49.4/72 (68.6%) | GF2 |

Greedy gets the highest reward because the reward function favors improvement trajectories (finding SGD first, then improving to GF2). Oracle gets the highest discovery score because it finds the best method first and explores systematically. Nobody hits 100% because `cross_challenge_analysis` (3 pts) requires the MultiChallengeEnv, and `correct_failure_classification` (16 pts) requires trying all failing methods.

## Discovery grading

The grader scores research quality across 12 categories, 72 points total. Each category measures whether the agent made a specific discovery, not just whether it got a good number.

| Category | Points | How to earn them |
|----------|--------|-----------------|
| Discovered algebraic solver | 10 | Try GF2, KM, or SMT and get 100% accuracy. Partial credit (3 pts) for trying but not solving. |
| Discovered KM influence | 7 | Solve with KM specifically. This is the O(n) approach vs O(C(n,k)) for Fourier. |
| Identified local learning failure | 5 | Try forward_forward and observe it fails (accuracy < 95%). This represents the finding that local learning rules can't detect k-th order interactions. |
| Found metric disagreement | 5 | Solve with both KM (ARD winner) and GF2 (DMC winner). Having both in the log means the agent has the data to notice rankings disagree. |
| Found curriculum speedup | 5 | Solve with curriculum learning. This represents discovering that training on small n first gives a speedup. |
| Identified parity invisibility | 5 | Observe at least 2 method failures and also find methods that work. The contrast reveals that parity is invisible to methods limited to low-order statistics. |
| Exploration breadth | 5 | Number of distinct methods that solve the problem (accuracy >= 95%). 1 point per method, max 5. |
| Efficiency | 5 | How quickly the best method was found. 5 pts if in steps 1-3, 4 pts in steps 4-6, decreasing to 0 at step 16+. |
| Optimized beyond baseline | 3 | Find any method with DMC below SGD baseline (1,278,460). GF2 at 8,607 easily qualifies. |
| Cross-challenge analysis | 3 | Only in MultiChallengeEnv. Solve methods across at least 2 different challenges (parity, sum, AND). |
| Cache model insight | 3 | Measure DMC across 3+ methods with different values. Having this spread means cache/energy behavior is observable. |
| Correct failure classification | 2/method | For each method tried that failed: 1 pt for observing the failure, 2 pts if the agent tried other methods afterward. Max 16 pts. |

Total: 72 points.

## Running the evaluation

```bash
PYTHONPATH=src python3 src/sparse_parity/eval/run_eval.py
```

Runs 3 baseline agents x 5 episodes in ~20 seconds (all 16 methods run live). Outputs results to `results/eval/baselines.json` and `results/eval/multi_challenge.json`.

## Answer key

The answer key at `src/sparse_parity/eval/answer_key.json` contains 36 experiments, 12 negative results, and the grading rubric. This is what makes the environment different from typical benchmarks: we know the optimal policy, so we can measure how close the agent gets.

## Portability

The Gymnasium interface is the standard. This environment can be adopted by:

- **PrimeIntellect**: Accepts Gymnasium envs for their research grants program. Our answer key and grading rubric are the differentiator.
- **Modal Labs**: Swap the harness to run experiments on GPU. Needed for scaling to larger n/k values.
- **Anthropic / OpenAI evals**: Wrap as tool-use evaluation. Agent gets `run_experiment` and `read_discoveries` tools instead of discrete indices.
- **HuggingFace Spaces**: Host a leaderboard where agents submit code and get scored.

The key advantage: we have ground truth. Most research envs don't know the optimal policy. We have 36 experiments showing what works, what fails, and why.

## Files

| File | Purpose |
|------|---------|
| `src/sparse_parity/eval/env.py` | Gymnasium environment |
| `src/sparse_parity/eval/baselines.py` | Random, Greedy, Oracle agents |
| `src/sparse_parity/eval/grader.py` | Discovery scoring |
| `src/sparse_parity/eval/answer_key.json` | Ground truth (36 experiments) |
| `src/sparse_parity/eval/README.md` | Full interface spec |
| `src/sparse_parity/eval/run_eval.py` | Evaluation script |

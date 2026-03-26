"""
Baseline agents for the SutroYaro Gymnasium environment.

Three agents of increasing sophistication:

    RandomAgent   -- picks a random method each step
    GreedyAgent   -- tries each method once in order, then repeats the best
    OracleAgent   -- uses answer_key.json to pick optimal methods first

Usage:
    from sparse_parity.eval.baselines import RandomAgent, GreedyAgent, OracleAgent

    agent = GreedyAgent()
    agent.reset(obs, info)
    action = agent.act(obs)
"""

import json
import os
import numpy as np

from sparse_parity.eval.env import METHOD_MAP, NUM_METHODS


class RandomAgent:
    """Pick a random method each step."""

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.name = "RandomAgent"

    def reset(self, obs, info):
        pass

    def act(self, obs):
        return int(self.rng.randint(NUM_METHODS))


class GreedyAgent:
    """
    Try each method once in order (0..15), then repeat the best.

    "Best" = method that solved the problem (acc >= 0.95) with the
    lowest value of the target metric. If no method has solved it,
    keep exploring in order.
    """

    def __init__(self):
        self.name = "GreedyAgent"
        self._next_untried = 0
        self._best_action = None
        self._best_score = float("inf")
        self._tried = set()

    def reset(self, obs, info):
        self._next_untried = 0
        self._best_action = None
        self._best_score = float("inf")
        self._tried = set()
        self._metric = info.get("metric", "dmc")

    def act(self, obs):
        # If we have tried all methods, repeat the best
        if self._next_untried >= NUM_METHODS:
            if self._best_action is not None:
                return self._best_action
            # Nothing solved -- just wrap around
            return 0

        # Update best from last result
        last = obs.get("last_result", {})
        if last.get("solved", 0) == 1:
            method_idx = last["method_index"]
            score_val = float(last.get(self._metric, np.array([float("inf")]))[0])
            if score_val < self._best_score:
                self._best_score = score_val
                self._best_action = method_idx

        # Try the next untried method
        action = self._next_untried
        self._next_untried += 1
        return action


class OracleAgent:
    """
    Uses answer_key.json to pick the method with the lowest DMC/ARD first.

    Looks up all experiments for the given challenge, filters to those with
    accuracy >= 0.95, sorts by the target metric, and picks them in order.
    Falls back to sequential order for methods not in the answer key.
    """

    def __init__(self):
        self.name = "OracleAgent"
        self._action_queue = []
        self._step = 0

    def reset(self, obs, info):
        self._step = 0
        challenge = info.get("challenge", "sparse-parity")
        metric = info.get("metric", "dmc")

        # Load answer key
        answer_key_path = os.path.join(
            os.path.dirname(__file__), "answer_key.json"
        )
        with open(answer_key_path) as f:
            data = json.load(f)

        # Find experiments for this challenge that solved the problem
        solved_experiments = []
        for exp in data["experiments"]:
            if exp.get("challenge") != challenge:
                continue
            acc = exp.get("accuracy", 0.0)
            if acc is None or acc < 0.95:
                continue
            score = exp.get(metric)
            if score is None:
                continue
            method = exp.get("method")
            if method in METHOD_MAP:
                solved_experiments.append((score, METHOD_MAP.index(method), method))

        # Sort by metric (lower is better), deduplicate by method
        solved_experiments.sort(key=lambda x: x[0])
        seen_methods = set()
        self._action_queue = []
        for score, action_idx, method in solved_experiments:
            if method not in seen_methods:
                self._action_queue.append(action_idx)
                seen_methods.add(method)

        # Append remaining methods that were not in the answer key
        for i in range(NUM_METHODS):
            if METHOD_MAP[i] not in seen_methods:
                self._action_queue.append(i)

    def act(self, obs):
        if self._step < len(self._action_queue):
            action = self._action_queue[self._step]
        else:
            # Exhausted the queue -- pick 0 (sgd) as fallback
            action = 0
        self._step += 1
        return action

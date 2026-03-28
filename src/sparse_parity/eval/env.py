"""
Gymnasium environment for SutroYaro method-selection task.

See README.md in this directory for the full interface specification.

Usage:
    import gymnasium as gym
    import sparse_parity.eval  # triggers registration (via __init__.py)

    env = gym.make("SutroYaro/SparseParity-v0",
        challenge="sparse-parity", n_bits=20, k_sparse=3,
        metric="dmc", budget=20, seed=42)

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(5)  # try gf2
"""

import math
import numpy as np
import gymnasium
from gymnasium import spaces

from sparse_parity.eval.backends import get_backend
from sparse_parity.eval import registry


# ---------------------------------------------------------------------------
# Backward-compatible module-level names.
#
# Other modules (baselines.py, run_eval.py) import METHOD_MAP, NUM_METHODS,
# and CHALLENGE_MAP from here.  These are now thin wrappers around the
# registry so that newly registered challenges/methods are visible
# everywhere without editing this file.
# ---------------------------------------------------------------------------

class _RegistryListProxy:
    """List-like proxy that always reflects the current registry state.

    Supports indexing, ``in``, ``len``, ``index``, iteration, and
    ``list()`` conversion so existing call sites keep working.
    """

    def __init__(self, list_fn):
        self._list_fn = list_fn

    def __getitem__(self, idx):
        return self._list_fn()[idx]

    def __contains__(self, item):
        return item in self._list_fn()

    def __len__(self):
        return len(self._list_fn())

    def __iter__(self):
        return iter(self._list_fn())

    def __repr__(self):
        return repr(self._list_fn())

    def index(self, value):
        return self._list_fn().index(value)


METHOD_MAP = _RegistryListProxy(registry.list_methods)
CHALLENGE_MAP = _RegistryListProxy(registry.list_challenges)
METRIC_MAP = ["ard", "dmc"]

# NUM_METHODS: computed from the registry.  External consumers that do
# ``from env import NUM_METHODS`` at import time get a snapshot; internal
# uses in the env classes call ``len(registry.list_methods())`` instead.
NUM_METHODS = 0  # will be set by __init__.py after default_registry runs


def _num_methods():
    """Live method count from the registry."""
    return len(registry.list_methods())


class SutroYaroEnv(gymnasium.Env):
    """
    Method-selection environment for energy-efficient sparse learning.

    The agent picks which method to try (discrete action 0..15).
    Each step runs the real harness and returns energy metrics.
    The goal is to find the method with the lowest ARD or DMC
    within a fixed budget of experiments.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        challenge="sparse-parity",
        n_bits=20,
        k_sparse=3,
        metric="dmc",
        budget=20,
        seed=42,
        harness_timeout=10.0,
        backend="local",
        render_mode=None,
    ):
        super().__init__()

        assert challenge in CHALLENGE_MAP, f"Unknown challenge: {challenge}"
        assert metric in METRIC_MAP, f"Unknown metric: {metric}"

        self.challenge = challenge
        self.n_bits = n_bits
        self.k_sparse = k_sparse
        self.metric = metric
        self.budget = budget
        self._seed = seed
        self.harness_timeout = harness_timeout
        self.render_mode = render_mode

        # Compute backend
        if isinstance(backend, str):
            self.backend = get_backend(backend, timeout=harness_timeout) if backend == "local" else get_backend(backend)
        else:
            # Accept a pre-built HarnessBackend instance
            self.backend = backend

        # Indices for observation encoding
        self._challenge_idx = CHALLENGE_MAP.index(challenge)
        self._metric_idx = METRIC_MAP.index(metric)

        # Number of methods / challenges from the registry (live)
        n_methods = _num_methods()
        n_challenges = len(registry.list_challenges())

        # ---- Spaces (match README spec exactly) ----
        self.observation_space = spaces.Dict({
            "challenge": spaces.Discrete(n_challenges),
            "n_bits": spaces.Discrete(101, start=3),
            "k_sparse": spaces.Discrete(11, start=3),
            "metric": spaces.Discrete(2),
            "best_score": spaces.Box(
                low=0.0, high=1e12, shape=(1,), dtype=np.float32
            ),
            "budget_remaining": spaces.Discrete(101),
            "steps_taken": spaces.Discrete(101),
            "methods_tried": spaces.MultiBinary(n_methods),
            "last_result": spaces.Dict({
                "method_index": spaces.Discrete(n_methods),
                "accuracy": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "ard": spaces.Box(0.0, 1e12, shape=(1,), dtype=np.float32),
                "dmc": spaces.Box(0.0, 1e12, shape=(1,), dtype=np.float32),
                "time_s": spaces.Box(0.0, 1e6, shape=(1,), dtype=np.float32),
                "solved": spaces.Discrete(2),
            }),
        })

        self.action_space = spaces.Discrete(n_methods)

        # Episode state (initialized in reset)
        self.steps_taken = 0
        self.best_score = float("inf")
        self.methods_tried = np.zeros(n_methods, dtype=np.int8)
        self.last_result_obs = self._empty_last_result()
        self.experiment_log = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        self.steps_taken = 0
        self.best_score = float("inf")
        self.methods_tried = np.zeros(_num_methods(), dtype=np.int8)
        self.last_result_obs = self._empty_last_result()
        self.experiment_log = []

        obs = self._build_obs()
        info = {
            "challenge": self.challenge,
            "n_bits": self.n_bits,
            "k_sparse": self.k_sparse,
            "metric": self.metric,
            "budget": self.budget,
            "methods": list(METHOD_MAP),
        }
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        method_name = METHOD_MAP[action]
        previous_best = self.best_score

        # Run the harness via backend
        result = self.backend.run(
            challenge=self.challenge,
            method=method_name,
            n_bits=self.n_bits,
            k_sparse=self.k_sparse,
            seed=self._seed,
        )

        # Update methods_tried
        self.methods_tried[action] = 1

        # Extract metrics
        accuracy = result.get("accuracy", 0.0)
        if accuracy is None:
            accuracy = 0.0
        ard = result.get("ard")
        dmc = result.get("dmc")
        time_s = result.get("time_s", 0.0)
        if time_s is None:
            time_s = 0.0
        error = result.get("error")
        found_secret = result.get("found_secret")
        solved = accuracy >= 0.95

        # Get the score for our target metric
        score = result.get(self.metric)

        # Update best_score if this is a valid solution and improved
        is_new_best = False
        improvement = 0.0
        if solved and score is not None:
            if score < self.best_score:
                is_new_best = True
                if self.best_score != float("inf"):
                    improvement = (self.best_score - score) / self.best_score
                self.best_score = score

        # Compute reward
        reward = self._compute_reward(
            accuracy=accuracy,
            score=score,
            previous_best=previous_best,
        )

        # Update last_result observation
        self.last_result_obs = {
            "method_index": action,
            "accuracy": np.array([accuracy], dtype=np.float32),
            "ard": np.array([ard if ard is not None else 0.0], dtype=np.float32),
            "dmc": np.array([dmc if dmc is not None else 0.0], dtype=np.float32),
            "time_s": np.array([time_s], dtype=np.float32),
            "solved": 1 if solved else 0,
        }

        # Advance step counter
        self.steps_taken += 1

        # Log the experiment
        self.experiment_log.append({
            "step": self.steps_taken,
            "method": method_name,
            "accuracy": accuracy,
            "ard": ard,
            "dmc": dmc,
            "time_s": time_s,
            "reward": reward,
            "is_new_best": is_new_best,
            "error": error,
        })

        # Termination
        terminated = False
        truncated = self.steps_taken >= self.budget

        obs = self._build_obs()
        info = {
            "method": method_name,
            "accuracy": accuracy,
            "ard": ard,
            "dmc": dmc,
            "time_s": time_s,
            "total_floats": result.get("total_floats"),
            "challenge": self.challenge,
            "found_secret": found_secret,
            "is_new_best": is_new_best,
            "improvement": improvement,
            "error": error,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if not self.experiment_log:
            print("No experiments run yet.")
            return

        print(f"\n{'='*70}")
        print(f"  Challenge: {self.challenge}  |  n={self.n_bits}  k={self.k_sparse}")
        print(f"  Metric: {self.metric}  |  Best: {self.best_score}")
        print(f"  Steps: {self.steps_taken}/{self.budget}")
        print(f"{'='*70}")
        print(f"  {'Step':>4}  {'Method':<18}  {'Acc':>5}  {'ARD':>10}  {'DMC':>12}  {'Reward':>7}  {'Best?'}")
        print(f"  {'-'*4}  {'-'*18}  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*7}  {'-'*5}")

        for entry in self.experiment_log:
            ard_str = f"{entry['ard']:>10.1f}" if entry["ard"] is not None else "       N/A"
            dmc_str = f"{entry['dmc']:>12.1f}" if entry["dmc"] is not None else "         N/A"
            best_str = "  *" if entry["is_new_best"] else ""
            print(
                f"  {entry['step']:>4}  {entry['method']:<18}  "
                f"{entry['accuracy']:>5.2f}  {ard_str}  {dmc_str}  "
                f"{entry['reward']:>7.3f}{best_str}"
            )

        print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self):
        best_for_obs = min(self.best_score, 1e12)
        if self.best_score == float("inf"):
            best_for_obs = 1e12

        return {
            "challenge": self._challenge_idx,
            "n_bits": self.n_bits,
            "k_sparse": self.k_sparse,
            "metric": self._metric_idx,
            "best_score": np.array([best_for_obs], dtype=np.float32),
            "budget_remaining": max(0, self.budget - self.steps_taken),
            "steps_taken": self.steps_taken,
            "methods_tried": self.methods_tried.copy(),
            "last_result": dict(self.last_result_obs),
        }

    def _empty_last_result(self):
        return {
            "method_index": 0,
            "accuracy": np.array([0.0], dtype=np.float32),
            "ard": np.array([0.0], dtype=np.float32),
            "dmc": np.array([0.0], dtype=np.float32),
            "time_s": np.array([0.0], dtype=np.float32),
            "solved": 0,
        }

    def _compute_reward(self, accuracy, score, previous_best):
        """Reward function per README spec."""
        if accuracy < 0.95:
            return -0.1

        if score is None:
            return 0.0

        if previous_best == float("inf"):
            # First successful solve
            return 10.0 / (1.0 + math.log10(max(score, 1.0)))

        if score < previous_best:
            # Improvement
            improvement_ratio = (previous_best - score) / previous_best
            return 10.0 * improvement_ratio

        # No improvement
        return -0.01

class MultiChallengeEnv(gymnasium.Env):
    """
    Cycles through multiple challenges, one per episode.

    Each reset advances to the next challenge in the list. After all
    challenges have been used, the cycle wraps around. Within each
    episode, the behavior is identical to SutroYaroEnv for that challenge.

    The observation space is the same as SutroYaroEnv. The ``info`` dict
    returned by reset/step includes ``cycle_index`` and ``challenges``
    so the agent can track progress across the full challenge cycle.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        challenges=None,
        budget_per=10,
        n_bits=20,
        k_sparse=3,
        metric="dmc",
        seed=42,
        harness_timeout=10.0,
        backend="local",
        render_mode=None,
    ):
        super().__init__()

        if challenges is None:
            challenges = list(CHALLENGE_MAP)  # all three
        for c in challenges:
            assert c in CHALLENGE_MAP, f"Unknown challenge: {c}"

        self.challenges = challenges
        self.budget_per = budget_per
        self.n_bits = n_bits
        self.k_sparse = k_sparse
        self.metric = metric
        self._seed = seed
        self.harness_timeout = harness_timeout
        self.render_mode = render_mode

        # Cycle state
        self._cycle_index = 0  # which challenge we are on (wraps)

        # The inner env does the real work
        self._inner = SutroYaroEnv(
            challenge=self.challenges[0],
            n_bits=n_bits,
            k_sparse=k_sparse,
            metric=metric,
            budget=budget_per,
            seed=seed,
            harness_timeout=harness_timeout,
            backend=backend,
            render_mode=render_mode,
        )

        # Mirror spaces from inner env
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        # Determine current challenge
        challenge = self.challenges[self._cycle_index % len(self.challenges)]

        # Reconfigure the inner env for this challenge
        self._inner.challenge = challenge
        self._inner._challenge_idx = CHALLENGE_MAP.index(challenge)
        self._inner.budget = self.budget_per

        if seed is not None:
            self._seed = seed

        obs, info = self._inner.reset(seed=seed, options=options)

        # Add multi-challenge context to info
        info["cycle_index"] = self._cycle_index
        info["challenges"] = self.challenges
        info["challenges_completed"] = self._cycle_index

        # Advance cycle for next reset
        self._cycle_index += 1

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(action)
        info["cycle_index"] = self._cycle_index - 1  # current (already advanced)
        info["challenges"] = self.challenges
        return obs, reward, terminated, truncated, info

    def render(self):
        self._inner.render()

    @property
    def experiment_log(self):
        return self._inner.experiment_log

    def close(self):
        self._inner.close()


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------
gymnasium.register(
    id="SutroYaro/SparseParity-v0",
    entry_point="sparse_parity.eval.env:SutroYaroEnv",
)

gymnasium.register(
    id="SutroYaro/MultiChallenge-v0",
    entry_point="sparse_parity.eval.env:MultiChallengeEnv",
)

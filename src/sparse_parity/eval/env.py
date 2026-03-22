"""
Gymnasium environment for SutroYaro method-selection task.

See README.md in this directory for the full interface specification.

Usage:
    import gymnasium as gym
    import sparse_parity.eval.env  # triggers registration

    env = gym.make("SutroYaro/SparseParity-v0",
        challenge="sparse-parity", n_bits=20, k_sparse=3,
        metric="dmc", budget=20, seed=42)

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(5)  # try gf2
"""

import math
import signal
import numpy as np
import gymnasium
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Fixed method index mapping (from search_space.yaml, v0)
# ---------------------------------------------------------------------------
METHOD_MAP = [
    "sgd",              # 0
    "perlayer",         # 1
    "sign_sgd",         # 2
    "curriculum",       # 3
    "forward_forward",  # 4
    "gf2",              # 5
    "km",               # 6
    "smt",              # 7
    "fourier",          # 8
    "lasso",            # 9
    "mdl",              # 10
    "mutual_info",      # 11
    "random_proj",      # 12
    "rl",               # 13
    "genetic_prog",     # 14
    "evolutionary",     # 15
]

NUM_METHODS = len(METHOD_MAP)

CHALLENGE_MAP = ["sparse-parity", "sparse-sum", "sparse-and"]
METRIC_MAP = ["ard", "dmc"]


class _HarnessTimeout(Exception):
    """Raised when a harness call exceeds its time budget."""


def _timeout_handler(signum, frame):
    raise _HarnessTimeout("Harness call timed out")


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

        # Indices for observation encoding
        self._challenge_idx = CHALLENGE_MAP.index(challenge)
        self._metric_idx = METRIC_MAP.index(metric)

        # ---- Spaces (match README spec exactly) ----
        self.observation_space = spaces.Dict({
            "challenge": spaces.Discrete(3),
            "n_bits": spaces.Discrete(101, start=3),
            "k_sparse": spaces.Discrete(11, start=3),
            "metric": spaces.Discrete(2),
            "best_score": spaces.Box(
                low=0.0, high=1e12, shape=(1,), dtype=np.float32
            ),
            "budget_remaining": spaces.Discrete(101),
            "steps_taken": spaces.Discrete(101),
            "methods_tried": spaces.MultiBinary(NUM_METHODS),
            "last_result": spaces.Dict({
                "method_index": spaces.Discrete(NUM_METHODS),
                "accuracy": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "ard": spaces.Box(0.0, 1e12, shape=(1,), dtype=np.float32),
                "dmc": spaces.Box(0.0, 1e12, shape=(1,), dtype=np.float32),
                "time_s": spaces.Box(0.0, 1e6, shape=(1,), dtype=np.float32),
                "solved": spaces.Discrete(2),
            }),
        })

        self.action_space = spaces.Discrete(NUM_METHODS)

        # Episode state (initialized in reset)
        self.steps_taken = 0
        self.best_score = float("inf")
        self.methods_tried = np.zeros(NUM_METHODS, dtype=np.int8)
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
        self.methods_tried = np.zeros(NUM_METHODS, dtype=np.int8)
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

        # Run the harness
        result = self._call_harness(method_name)

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

    def _call_harness(self, method_name):
        """Call the harness with timeout handling."""
        import harness  # src/ must be on PYTHONPATH

        measure_fn = {
            "sparse-parity": harness.measure_sparse_parity,
            "sparse-sum": harness.measure_sparse_sum,
            "sparse-and": harness.measure_sparse_and,
        }[self.challenge]

        try:
            # Set timeout (Unix only, graceful)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(self.harness_timeout))

            result = measure_fn(
                method=method_name,
                n_bits=self.n_bits,
                k_sparse=self.k_sparse,
                seed=self._seed,
            )

            signal.alarm(0)  # cancel alarm
            signal.signal(signal.SIGALRM, old_handler)

            # If the harness returned an error dict, treat as failure
            if "error" in result and result.get("accuracy") is None:
                result.setdefault("accuracy", 0.0)

            return result

        except _HarnessTimeout:
            signal.alarm(0)
            return {
                "accuracy": 0.0,
                "ard": None,
                "dmc": None,
                "time_s": self.harness_timeout,
                "total_floats": None,
                "error": f"Method '{method_name}' timed out after {self.harness_timeout}s",
                "method": method_name,
            }
        except Exception as e:
            signal.alarm(0)
            return {
                "accuracy": 0.0,
                "ard": None,
                "dmc": None,
                "time_s": None,
                "total_floats": None,
                "error": f"Method '{method_name}' raised: {type(e).__name__}: {e}",
                "method": method_name,
            }


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------
gymnasium.register(
    id="SutroYaro/SparseParity-v0",
    entry_point="sparse_parity.eval.env:SutroYaroEnv",
)

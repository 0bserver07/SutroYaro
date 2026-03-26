#!/usr/bin/env python3
"""
Demo: Can an agent find the cheapest method to solve sparse parity?

The Sutro Group constraint: solve within 1980s compute budgets.
Under 1 second, ideally under 10ms. The question is not whether
the problem can be solved (it can, many ways), but which method
moves the least data doing it.

This demo walks through what the eval environment tests:
1. An agent picks methods and observes energy metrics
2. Some methods solve it, some fail -- both are signal
3. The agent gets graded on what it figured out, not just the best number

Run: PYTHONPATH=src python3 src/sparse_parity/eval/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sparse_parity.eval.env import SutroYaroEnv
from sparse_parity.eval.grader import DiscoveryGrader
from sparse_parity.eval.registry import list_methods


def demo_step_by_step():
    """Walk through the environment step by step."""
    print("=" * 70)
    print("  SUTRO EVAL: Can an agent find energy-efficient solutions?")
    print("=" * 70)
    print()
    print("  Constraint: solve sparse parity within 1980s compute budgets.")
    print("  16 methods available. Some solve it in 509 microseconds.")
    print("  Some fail entirely. An agent has to figure out which is which.")
    print()

    env = SutroYaroEnv(challenge="sparse-parity", metric="dmc", budget=16)
    env.reset()

    # Simulate an agent's research trajectory
    # Start with the obvious choice (SGD), then explore
    trajectory = [
        (0, "Start with SGD -- the standard approach"),
        (4, "Try Forward-Forward -- does local learning work?"),
        (5, "Try GF(2) -- algebraic solver"),
        (6, "Try KM -- influence estimation"),
        (3, "Try Curriculum -- train small, expand"),
        (8, "Try Fourier -- exhaustive Walsh-Hadamard"),
    ]

    print("  Step  Method               Acc     DMC              Reward  Note")
    print("  ----  -------------------  -----   ---------------  ------  ----")

    for step, (action, note) in enumerate(trajectory, 1):
        method = list_methods()[action]
        obs, reward, _, truncated, info = env.step(action)
        acc = info.get("accuracy", 0)
        dmc = info.get("dmc")
        dmc_str = f"{dmc:>15,.0f}" if dmc and dmc > 0 else "            N/A"
        best = " *" if info.get("is_new_best") else ""

        print(f"  {step:>4}  {method:<20s} {acc:<5.2f}  {dmc_str}  {reward:>6.2f}  {note}{best}")

    print()
    env.render()

    # Grade the trajectory
    grader = DiscoveryGrader()
    report = grader.grade(env.experiment_log)

    print()
    print("=" * 70)
    print("  DISCOVERY GRADE")
    print("=" * 70)
    print()
    print(f"  Score: {report.total_score}/{report.max_possible} ({report.percentage:.0f}%)")
    print()

    for name, cat in report.categories.items():
        marker = "+" if cat["score"] > 0 else " "
        print(f"  {marker} {name}: {cat['score']}/{cat['max']}")
        if cat["score"] > 0:
            # Show why points were awarded
            detail = cat["details"]
            if len(detail) > 80:
                detail = detail[:77] + "..."
            print(f"      {detail}")

    print()
    print("  What this means:")
    print(f"  The agent tried {len(trajectory)} methods and scored {report.percentage:.0f}%.")
    print(f"  It found GF(2) (DMC 8,607) which is 149x cheaper than SGD (DMC 1,278,460).")
    print(f"  It observed Forward-Forward failing, which reveals the problem structure.")
    print(f"  A perfect score requires trying all 16 methods and using MultiChallengeEnv.")


def demo_add_method():
    """Show how to add a new method to the environment."""
    print()
    print("=" * 70)
    print("  ADDING A NEW METHOD")
    print("=" * 70)
    print()
    print("  The registry lets you add methods without editing env.py.")
    print()

    from sparse_parity.eval.registry import register_method, get_method_index

    # Show before
    before = len(list_methods())
    print(f"  Methods before: {before}")

    # Register a new method
    register_method(
        "my_solver",
        category="algebraic",
        applicable_challenges=["sparse-parity"],
        description="Example: a new algebraic solver",
    )

    after = len(list_methods())
    idx = get_method_index("my_solver")
    print(f"  Methods after:  {after}")
    print(f"  New method 'my_solver' at index {idx}")
    print()
    print("  To make it runnable, add a fallback in backends.py:")
    print("  FALLBACK_METHODS['my_solver'] = my_solver_function")
    print()
    print("  To add the ground truth, update answer_key.json:")
    print('  {"method": "my_solver", "accuracy": 1.0, "dmc": 500, ...}')
    print()
    print("  Full guide: docs/research/adding-an-eval-challenge.md")


def demo_speed_comparison():
    """Show which methods meet the 1980s compute constraint."""
    print()
    print("=" * 70)
    print("  SPEED CHECK: Which methods fit 1980s compute budgets?")
    print("=" * 70)
    print()
    print("  Constraint: under 1 second, ideally under 10ms.")
    print("  Metric: DMC (total data movement cost, lower = less energy)")
    print()

    env = SutroYaroEnv(challenge="sparse-parity", metric="dmc", budget=16)
    env.reset()

    results = []
    for i in range(16):
        obs, r, _, _, info = env.step(i)
        method = info.get("method", "?")
        acc = info.get("accuracy", 0)
        dmc = info.get("dmc")
        time_s = info.get("time_s", 0)
        results.append((method, acc, dmc, time_s))

    # Sort by DMC (solved methods only)
    solved = [(m, a, d, t) for m, a, d, t in results if a >= 0.95 and d and d > 0]
    solved.sort(key=lambda x: x[2])

    failed = [(m, a, d, t) for m, a, d, t in results if a < 0.95]

    print(f"  {'Method':<20s} {'Acc':>5s} {'DMC':>15s} {'Time':>8s}  Verdict")
    print(f"  {'---':<20s} {'---':>5s} {'---':>15s} {'---':>8s}  ---")

    for m, a, d, t in solved:
        if t and t < 0.01:
            verdict = "UNDER 10ms"
        elif t and t < 1.0:
            verdict = "under 1s"
        else:
            verdict = ""
        print(f"  {m:<20s} {a:>5.2f} {d:>15,.0f} {t:>7.3f}s  {verdict}")

    if failed:
        print()
        print("  Failed methods (accuracy < 95%):")
        for m, a, d, t in failed:
            print(f"  {m:<20s} {a:>5.2f}  -- fails on parity")

    print()
    under_10ms = sum(1 for _, _, _, t in solved if t and t < 0.01)
    under_1s = sum(1 for _, _, _, t in solved if t and t < 1.0)
    print(f"  {under_10ms} methods under 10ms, {under_1s} under 1 second.")
    print(f"  Best DMC: {solved[0][0]} at {solved[0][2]:,.0f}")
    print(f"  Worst DMC (that solves): {solved[-1][0]} at {solved[-1][2]:,.0f}")
    print(f"  Spread: {solved[-1][2] / solved[0][2]:,.0f}x")


if __name__ == "__main__":
    demo_step_by_step()
    demo_add_method()
    demo_speed_comparison()

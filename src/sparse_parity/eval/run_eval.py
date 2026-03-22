#!/usr/bin/env python3
"""
Evaluate baseline agents on the SutroYaro Gymnasium environment.

Usage:
    PYTHONPATH=src python3 src/sparse_parity/eval/run_eval.py

Runs each baseline agent for a number of episodes on sparse-parity,
reports summary statistics, and saves results to results/eval/baselines.json.
"""

import json
import os
import sys
import time

import numpy as np
import gymnasium as gym

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sparse_parity.eval.env  # register the environment
from sparse_parity.eval.baselines import RandomAgent, GreedyAgent, OracleAgent
from sparse_parity.eval.env import METHOD_MAP


def run_episode(env, agent, seed=None):
    """Run one episode, return summary dict."""
    obs, info = env.reset(seed=seed)
    agent.reset(obs, info)

    total_reward = 0.0
    steps = 0
    best_method = None
    best_score = float("inf")
    methods_discovered = []  # methods that solved the problem

    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if info.get("accuracy", 0.0) >= 0.95:
            method_name = info.get("method", METHOD_MAP[action])
            if method_name not in methods_discovered:
                methods_discovered.append(method_name)

            metric_val = info.get(env.unwrapped.metric)
            if metric_val is not None and metric_val < best_score:
                best_score = metric_val
                best_method = method_name

        if terminated or truncated:
            break

    return {
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "best_method": best_method,
        "best_score": best_score if best_score < float("inf") else None,
        "methods_discovered": methods_discovered,
        "n_discovered": len(methods_discovered),
    }


def evaluate_agent(agent, env_kwargs, n_episodes=5, seeds=None):
    """Run n_episodes and return aggregate stats."""
    if seeds is None:
        seeds = list(range(42, 42 + n_episodes))

    episodes = []
    for i, seed in enumerate(seeds):
        env = gym.make("SutroYaro/SparseParity-v0", **env_kwargs)
        result = run_episode(env, agent, seed=seed)
        episodes.append(result)
        env.close()

    # Aggregate
    rewards = [e["total_reward"] for e in episodes]
    n_discovered = [e["n_discovered"] for e in episodes]
    all_discovered = set()
    for e in episodes:
        all_discovered.update(e["methods_discovered"])

    return {
        "agent": agent.name,
        "n_episodes": n_episodes,
        "mean_reward": round(float(np.mean(rewards)), 4),
        "std_reward": round(float(np.std(rewards)), 4),
        "min_reward": round(float(np.min(rewards)), 4),
        "max_reward": round(float(np.max(rewards)), 4),
        "mean_discovered": round(float(np.mean(n_discovered)), 2),
        "all_methods_found": sorted(all_discovered),
        "best_methods": [e["best_method"] for e in episodes],
        "best_scores": [e["best_score"] for e in episodes],
        "episodes": episodes,
    }


def print_summary(results):
    """Print a summary table to stdout."""
    print()
    print("=" * 78)
    print("  SutroYaro Baseline Evaluation Results")
    print("=" * 78)
    print()

    # Header
    print(f"  {'Agent':<16}  {'Mean Reward':>12}  {'Std':>7}  "
          f"{'Mean Found':>10}  {'Best Method':<12}")
    print(f"  {'-'*16}  {'-'*12}  {'-'*7}  {'-'*10}  {'-'*12}")

    for r in results:
        # Most common best method
        from collections import Counter
        best_counts = Counter(r["best_methods"])
        top_method = best_counts.most_common(1)[0][0] if best_counts else "N/A"

        print(f"  {r['agent']:<16}  {r['mean_reward']:>12.4f}  "
              f"{r['std_reward']:>7.4f}  {r['mean_discovered']:>10.2f}  "
              f"{top_method:<12}")

    print()

    # Detail per agent
    for r in results:
        print(f"  --- {r['agent']} ---")
        print(f"  Methods found across all episodes: {', '.join(r['all_methods_found'])}")
        for i, ep in enumerate(r["episodes"]):
            print(f"    Episode {i+1}: reward={ep['total_reward']:>8.4f}  "
                  f"steps={ep['steps']:>2}  best={ep['best_method']!s:<8}  "
                  f"score={ep['best_score']}  "
                  f"discovered={ep['methods_discovered']}")
        print()


def main():
    # Configuration
    n_episodes = 5
    budget = 16  # enough to try all 16 methods once
    challenge = "sparse-parity"
    metric = "dmc"
    n_bits = 20
    k_sparse = 3

    env_kwargs = {
        "challenge": challenge,
        "n_bits": n_bits,
        "k_sparse": k_sparse,
        "metric": metric,
        "budget": budget,
        "harness_timeout": 10.0,
    }

    seeds = [42, 123, 456, 789, 1337]

    agents = [
        RandomAgent(seed=0),
        GreedyAgent(),
        OracleAgent(),
    ]

    print(f"\nEvaluating {len(agents)} agents x {n_episodes} episodes "
          f"on {challenge} (n={n_bits}, k={k_sparse}, metric={metric}, budget={budget})")
    print(f"Seeds: {seeds}")
    print()

    all_results = []
    t0 = time.time()

    for agent in agents:
        print(f"  Running {agent.name}...", end="", flush=True)
        t_agent = time.time()
        result = evaluate_agent(agent, env_kwargs, n_episodes=n_episodes, seeds=seeds)
        elapsed = time.time() - t_agent
        print(f" done ({elapsed:.1f}s)")
        all_results.append(result)

    total_time = time.time() - t0
    print(f"\nTotal evaluation time: {total_time:.1f}s")

    # Print summary
    print_summary(all_results)

    # Save results
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    results_dir = os.path.join(project_root, "results", "eval")
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, "baselines.json")
    output = {
        "eval_config": {
            "challenge": challenge,
            "n_bits": n_bits,
            "k_sparse": k_sparse,
            "metric": metric,
            "budget": budget,
            "n_episodes": n_episodes,
            "seeds": seeds,
        },
        "total_time_s": round(total_time, 2),
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

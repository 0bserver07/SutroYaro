#!/usr/bin/env python3
"""
Evaluate baseline agents on the SutroYaro Gymnasium environment.

Usage:
    PYTHONPATH=src python3 src/sparse_parity/eval/run_eval.py

Runs each baseline agent for a number of episodes on sparse-parity,
reports summary statistics, and saves results to results/eval/baselines.json.

Also evaluates baselines on MultiChallengeEnv (cycling through sparse-parity,
sparse-sum, and sparse-and) and saves results to results/eval/multi_challenge.json.
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
from sparse_parity.eval.env import METHOD_MAP, CHALLENGE_MAP
from sparse_parity.eval.grader import DiscoveryGrader


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

    # Capture the experiment log from the unwrapped env for grading
    experiment_log = list(env.unwrapped.experiment_log)

    return {
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "best_method": best_method,
        "best_score": best_score if best_score < float("inf") else None,
        "methods_discovered": methods_discovered,
        "n_discovered": len(methods_discovered),
        "experiment_log": experiment_log,
    }


def evaluate_agent(agent, env_kwargs, n_episodes=5, seeds=None, grader=None):
    """Run n_episodes and return aggregate stats with optional grading."""
    if seeds is None:
        seeds = list(range(42, 42 + n_episodes))

    challenge = env_kwargs.get("challenge", "sparse-parity")
    episodes = []
    grade_reports = []

    for i, seed in enumerate(seeds):
        env = gym.make("SutroYaro/SparseParity-v0", **env_kwargs)
        result = run_episode(env, agent, seed=seed)
        episodes.append(result)

        # Grade the episode if a grader is provided
        if grader is not None:
            report = grader.grade(result["experiment_log"], challenge=challenge)
            grade_reports.append(report)
            result["grade"] = {
                "total_score": report.total_score,
                "max_possible": report.max_possible,
                "percentage": report.percentage,
                "categories": report.categories,
                "summary": report.summary,
            }

        env.close()

    # Aggregate
    rewards = [e["total_reward"] for e in episodes]
    n_discovered = [e["n_discovered"] for e in episodes]
    all_discovered = set()
    for e in episodes:
        all_discovered.update(e["methods_discovered"])

    result_dict = {
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

    # Add grading aggregates if grading was performed
    if grade_reports:
        scores = [r.total_score for r in grade_reports]
        percentages = [r.percentage for r in grade_reports]
        result_dict["grading"] = {
            "mean_score": round(float(np.mean(scores)), 2),
            "std_score": round(float(np.std(scores)), 2),
            "min_score": round(float(np.min(scores)), 2),
            "max_score": round(float(np.max(scores)), 2),
            "mean_percentage": round(float(np.mean(percentages)), 1),
            "max_possible": grade_reports[0].max_possible,
        }

    return result_dict


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


def print_grading_summary(results):
    """Print discovery grading results for each agent."""
    print()
    print("=" * 78)
    print("  Discovery Grading Results")
    print("=" * 78)
    print()

    # Summary table
    print(f"  {'Agent':<16}  {'Mean Score':>10}  {'Std':>6}  "
          f"{'Max Possible':>12}  {'Mean %':>7}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*6}  {'-'*12}  {'-'*7}")

    for r in results:
        g = r.get("grading")
        if g is None:
            continue
        print(f"  {r['agent']:<16}  {g['mean_score']:>10.2f}  "
              f"{g['std_score']:>6.2f}  {g['max_possible']:>12.0f}  "
              f"{g['mean_percentage']:>6.1f}%")

    print()

    # Per-agent category breakdown (use first episode as representative)
    for r in results:
        episodes = r.get("episodes", [])
        if not episodes or "grade" not in episodes[0]:
            continue

        print(f"  --- {r['agent']} (episode 1 breakdown) ---")
        grade = episodes[0]["grade"]
        for cat_name, cat in grade["categories"].items():
            marker = "+" if cat["score"] > 0 else " "
            print(f"    {marker} {cat_name}: {cat['score']}/{cat['max']}"
                  f"  -- {cat['details']}")
        print(f"    Summary: {grade['summary']}")
        print()


def run_multi_challenge_episode(env, agent, seed=None):
    """
    Run one full cycle of MultiChallengeEnv (one episode per challenge).

    Returns a list of per-challenge episode summaries.
    """
    challenges = env.unwrapped.challenges
    results_per_challenge = []

    for i, challenge in enumerate(challenges):
        obs, info = env.reset(seed=seed)
        agent.reset(obs, info)

        total_reward = 0.0
        steps = 0
        best_method = None
        best_score = float("inf")
        methods_discovered = []

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

        results_per_challenge.append({
            "challenge": challenge,
            "total_reward": round(total_reward, 4),
            "steps": steps,
            "best_method": best_method,
            "best_score": best_score if best_score < float("inf") else None,
            "methods_discovered": methods_discovered,
            "n_discovered": len(methods_discovered),
        })

    return results_per_challenge


def evaluate_agent_multi(agent, env_kwargs, n_cycles=3, seeds=None):
    """Run n_cycles of multi-challenge evaluation and return aggregate stats."""
    challenges = env_kwargs.get("challenges", list(CHALLENGE_MAP))
    if seeds is None:
        seeds = list(range(42, 42 + n_cycles))

    all_cycles = []
    for seed in seeds:
        env = gym.make("SutroYaro/MultiChallenge-v0", **env_kwargs)
        cycle_results = run_multi_challenge_episode(env, agent, seed=seed)
        all_cycles.append(cycle_results)
        env.close()

    # Aggregate per challenge
    per_challenge = {}
    for challenge in challenges:
        challenge_episodes = []
        for cycle in all_cycles:
            for ep in cycle:
                if ep["challenge"] == challenge:
                    challenge_episodes.append(ep)

        rewards = [e["total_reward"] for e in challenge_episodes]
        n_discovered = [e["n_discovered"] for e in challenge_episodes]
        all_discovered = set()
        for e in challenge_episodes:
            all_discovered.update(e["methods_discovered"])

        per_challenge[challenge] = {
            "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
            "std_reward": round(float(np.std(rewards)), 4) if rewards else 0.0,
            "mean_discovered": round(float(np.mean(n_discovered)), 2) if n_discovered else 0.0,
            "all_methods_found": sorted(all_discovered),
            "best_methods": [e["best_method"] for e in challenge_episodes],
            "best_scores": [e["best_score"] for e in challenge_episodes],
            "episodes": challenge_episodes,
        }

    # Overall aggregate
    all_rewards = [e["total_reward"] for cycle in all_cycles for e in cycle]

    return {
        "agent": agent.name,
        "n_cycles": n_cycles,
        "challenges": challenges,
        "overall_mean_reward": round(float(np.mean(all_rewards)), 4) if all_rewards else 0.0,
        "overall_std_reward": round(float(np.std(all_rewards)), 4) if all_rewards else 0.0,
        "per_challenge": per_challenge,
    }


def print_multi_summary(results):
    """Print a summary table for multi-challenge evaluation."""
    print()
    print("=" * 78)
    print("  SutroYaro Multi-Challenge Evaluation Results")
    print("=" * 78)
    print()

    # Overall table
    print(f"  {'Agent':<16}  {'Overall Reward':>14}  {'Std':>7}")
    print(f"  {'-'*16}  {'-'*14}  {'-'*7}")

    for r in results:
        print(f"  {r['agent']:<16}  {r['overall_mean_reward']:>14.4f}  "
              f"{r['overall_std_reward']:>7.4f}")

    print()

    # Per-challenge breakdown
    for r in results:
        print(f"  --- {r['agent']} ---")
        for challenge, stats in r["per_challenge"].items():
            from collections import Counter
            best_counts = Counter(stats["best_methods"])
            top_method = best_counts.most_common(1)[0][0] if best_counts else "N/A"
            print(f"    {challenge:<16}  reward={stats['mean_reward']:>8.4f}  "
                  f"found={stats['mean_discovered']:.1f}  "
                  f"best={top_method}  "
                  f"methods={', '.join(stats['all_methods_found']) or 'none'}")
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

    # Initialize discovery grader
    grader = DiscoveryGrader()

    # ---------------------------------------------------------------
    # Part 1: Single-challenge evaluation (sparse-parity, existing)
    # ---------------------------------------------------------------
    print(f"\nEvaluating {len(agents)} agents x {n_episodes} episodes "
          f"on {challenge} (n={n_bits}, k={k_sparse}, metric={metric}, budget={budget})")
    print(f"Seeds: {seeds}")
    print()

    all_results = []
    t0 = time.time()

    for agent in agents:
        print(f"  Running {agent.name}...", end="", flush=True)
        t_agent = time.time()
        result = evaluate_agent(
            agent, env_kwargs, n_episodes=n_episodes, seeds=seeds, grader=grader
        )
        elapsed = time.time() - t_agent
        print(f" done ({elapsed:.1f}s)")
        all_results.append(result)

    total_time = time.time() - t0
    print(f"\nTotal evaluation time: {total_time:.1f}s")

    # Print summary
    print_summary(all_results)

    # Print grading summary
    print_grading_summary(all_results)

    # Save results
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    results_dir = os.path.join(project_root, "results", "eval")
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, "baselines.json")

    # Strip experiment_log from episodes before saving (too verbose for JSON)
    save_results = []
    for r in all_results:
        r_copy = dict(r)
        r_copy["episodes"] = [
            {k: v for k, v in ep.items() if k != "experiment_log"}
            for ep in r_copy["episodes"]
        ]
        save_results.append(r_copy)

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
        "results": save_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")

    # ---------------------------------------------------------------
    # Part 2: Multi-challenge evaluation
    # ---------------------------------------------------------------
    n_cycles = 3
    budget_per = 10
    multi_seeds = [42, 123, 456]

    multi_env_kwargs = {
        "budget_per": budget_per,
        "n_bits": n_bits,
        "k_sparse": k_sparse,
        "metric": metric,
        "harness_timeout": 10.0,
    }

    multi_agents = [
        RandomAgent(seed=0),
        GreedyAgent(),
        OracleAgent(),
    ]

    print(f"\n{'='*78}")
    print(f"  Multi-Challenge Evaluation")
    print(f"{'='*78}")
    print(f"\nEvaluating {len(multi_agents)} agents x {n_cycles} cycles "
          f"on all challenges (n={n_bits}, k={k_sparse}, metric={metric}, budget_per={budget_per})")
    print(f"Challenges: {list(CHALLENGE_MAP)}")
    print(f"Seeds: {multi_seeds}")
    print()

    multi_results = []
    t0_multi = time.time()

    for agent in multi_agents:
        print(f"  Running {agent.name} (multi)...", end="", flush=True)
        t_agent = time.time()
        result = evaluate_agent_multi(
            agent, multi_env_kwargs, n_cycles=n_cycles, seeds=multi_seeds
        )
        elapsed = time.time() - t_agent
        print(f" done ({elapsed:.1f}s)")
        multi_results.append(result)

    total_multi_time = time.time() - t0_multi
    print(f"\nMulti-challenge evaluation time: {total_multi_time:.1f}s")

    # Print multi-challenge summary
    print_multi_summary(multi_results)

    # Save multi-challenge results
    multi_output_path = os.path.join(results_dir, "multi_challenge.json")
    multi_output = {
        "eval_config": {
            "challenges": list(CHALLENGE_MAP),
            "n_bits": n_bits,
            "k_sparse": k_sparse,
            "metric": metric,
            "budget_per": budget_per,
            "n_cycles": n_cycles,
            "seeds": multi_seeds,
        },
        "total_time_s": round(total_multi_time, 2),
        "results": multi_results,
    }

    with open(multi_output_path, "w") as f:
        json.dump(multi_output, f, indent=2)

    print(f"Multi-challenge results saved to {multi_output_path}")


if __name__ == "__main__":
    main()

"""
Discovery grader for SutroYaro evaluation.

Scores an agent's trajectory based on what it figured out -- which methods
it tried, what results it observed, and what discoveries it made -- rather
than just metric improvement.

Usage:
    from sparse_parity.eval.grader import DiscoveryGrader

    grader = DiscoveryGrader()
    report = grader.grade(env.experiment_log, challenge="sparse-parity")
    print(report.summary)
    print(f"Score: {report.total_score}/{report.max_possible} ({report.percentage:.0f}%)")
"""

import json
import os
from dataclasses import dataclass


@dataclass
class GradeReport:
    total_score: float
    max_possible: float
    percentage: float
    categories: dict  # category_name -> {score, max, details}
    summary: str  # one-line description

    def __str__(self):
        lines = [
            f"Score: {self.total_score}/{self.max_possible} ({self.percentage:.0f}%)",
            f"Summary: {self.summary}",
            "",
        ]
        for name, cat in self.categories.items():
            marker = "+" if cat["score"] > 0 else " "
            lines.append(
                f"  {marker} {name}: {cat['score']}/{cat['max']}  -- {cat['details']}"
            )
        return "\n".join(lines)


# Methods classified as algebraic solvers in the answer key
ALGEBRAIC_METHODS = {"gf2", "km", "smt"}

# Methods classified as local learning (known to fail on parity)
LOCAL_LEARNING_METHODS = {
    "hebbian", "predictive_coding", "equilibrium_prop", "target_prop"
}

# Methods that fail on sparse-parity (acc < 0.95) in the 16-method action space.
# forward_forward: local learning can't coordinate multi-layer features (58.5%)
# perlayer: often times out before grokking triggers
# sign_sgd: conservative lr=0.01, often times out
# genetic_prog: needle-in-haystack, usually doesn't find the secret
KNOWN_FAILURE_METHODS_PARITY = {"forward_forward", "perlayer", "sign_sgd", "genetic_prog"}

# SGD baseline DMC for sparse-parity
SGD_BASELINE_DMC = 1_278_460

# Methods that win on ARD vs DMC (for metric disagreement detection)
ARD_WINNERS = {"km"}    # KM has better ARD than GF2
DMC_WINNERS = {"gf2"}   # GF2 has better DMC than KM


class DiscoveryGrader:
    """Grades an agent's experiment trajectory on discovery quality."""

    def __init__(self, answer_key_path=None):
        if answer_key_path is None:
            answer_key_path = os.path.join(
                os.path.dirname(__file__), "answer_key.json"
            )
        with open(answer_key_path) as f:
            self.answer_key = json.load(f)

        self._rubric = self.answer_key.get("grading_rubric", {})

    def grade(self, experiment_log, challenge="sparse-parity"):
        """
        Grade an experiment log (list of step dicts from env).

        Each entry has: step, method, accuracy, ard, dmc, time_s, reward,
        is_new_best, error.

        Returns a GradeReport with per-category scores.
        """
        categories = {}

        # Precompute useful sets from the log
        all_methods = [e["method"] for e in experiment_log]
        unique_methods = list(dict.fromkeys(all_methods))  # preserve order
        solved_methods = [
            e["method"] for e in experiment_log if e["accuracy"] >= 0.95
        ]
        solved_set = set(solved_methods)
        failed_methods = [
            e["method"] for e in experiment_log
            if e["accuracy"] < 0.95
        ]
        failed_set = set(failed_methods)

        # Build a dict: method -> best result entry (by target metric or accuracy)
        method_results = {}
        for e in experiment_log:
            m = e["method"]
            if m not in method_results:
                method_results[m] = e
            elif e["accuracy"] > method_results[m]["accuracy"]:
                method_results[m] = e

        # 1. discovered_algebraic_solver (10 pts)
        categories["discovered_algebraic_solver"] = self._grade_algebraic(
            experiment_log, solved_set, challenge
        )

        # 2. identified_local_learning_failure (5 pts)
        categories["identified_local_learning_failure"] = (
            self._grade_local_learning_failure(experiment_log, failed_set, challenge)
        )

        # 3. found_metric_disagreement (5 pts)
        categories["found_metric_disagreement"] = (
            self._grade_metric_disagreement(experiment_log, solved_set)
        )

        # 4. optimized_beyond_baseline (3 pts)
        categories["optimized_beyond_baseline"] = self._grade_beyond_baseline(
            experiment_log, challenge
        )

        # 5. correct_failure_classification (2 pts each, up to 16)
        categories["correct_failure_classification"] = (
            self._grade_failure_classification(
                experiment_log, failed_set, solved_set, unique_methods
            )
        )

        # 6. efficiency (up to 5 pts)
        categories["efficiency"] = self._grade_efficiency(experiment_log)

        # 7. exploration_breadth (up to 5 pts)
        categories["exploration_breadth"] = self._grade_breadth(
            experiment_log, solved_set
        )

        # 8. discovered_km_influence (7 pts)
        categories["discovered_km_influence"] = self._grade_km_influence(
            experiment_log, solved_set
        )

        # 9. found_curriculum_speedup (5 pts)
        categories["found_curriculum_speedup"] = self._grade_curriculum_speedup(
            experiment_log, solved_set
        )

        # 10. identified_parity_invisibility (5 pts)
        categories["identified_parity_invisibility"] = self._grade_parity_invisibility(
            experiment_log, failed_set
        )

        # 11. cross_challenge_analysis (3 pts)
        categories["cross_challenge_analysis"] = self._grade_cross_challenge(
            experiment_log, solved_set, challenge
        )

        # 12. cache_model_insight (3 pts)
        categories["cache_model_insight"] = self._grade_cache_insight(
            experiment_log, solved_set
        )

        # Compute totals
        total_score = sum(c["score"] for c in categories.values())
        max_possible = sum(c["max"] for c in categories.values())
        percentage = (total_score / max_possible * 100) if max_possible > 0 else 0.0

        # Summary
        n_solved = len(solved_set)
        n_tried = len(set(all_methods))
        best_method = None
        best_dmc = float("inf")
        for e in experiment_log:
            if e["accuracy"] >= 0.95 and e.get("dmc") is not None:
                if e["dmc"] < best_dmc:
                    best_dmc = e["dmc"]
                    best_method = e["method"]

        if best_method:
            summary = (
                f"{n_solved} methods solved out of {n_tried} tried, "
                f"best={best_method} (DMC={best_dmc:,.0f}), "
                f"score {total_score:.1f}/{max_possible:.0f}"
            )
        else:
            summary = (
                f"0 methods solved out of {n_tried} tried, "
                f"score {total_score:.1f}/{max_possible:.0f}"
            )

        return GradeReport(
            total_score=round(total_score, 2),
            max_possible=round(max_possible, 2),
            percentage=round(percentage, 2),
            categories=categories,
            summary=summary,
        )

    def grade_episode(self, env):
        """Grade using env's internal experiment log after episode ends."""
        return self.grade(env.experiment_log, challenge=env.challenge)

    # ------------------------------------------------------------------
    # Category graders
    # ------------------------------------------------------------------

    def _grade_km_influence(self, log, solved_set):
        """
        discovered_km_influence (7 pts):
        Found KM influence estimation -- O(n) not O(C(n,k)).
        """
        max_pts = 7
        if "km" in solved_set:
            details = "Solved with KM influence estimation (O(n) approach)"
            return {"score": max_pts, "max": max_pts, "details": details}
        elif any(e["method"] == "km" for e in log):
            details = "Tried KM but did not solve"
            return {"score": 2, "max": max_pts, "details": details}
        else:
            details = "Never tried KM influence estimation"
            return {"score": 0, "max": max_pts, "details": details}

    def _grade_curriculum_speedup(self, log, solved_set):
        """
        found_curriculum_speedup (5 pts):
        Discovered that curriculum (train small n first) gives speedup.
        """
        max_pts = 5
        if "curriculum" in solved_set:
            details = "Solved with curriculum learning (n-curriculum speedup)"
            return {"score": max_pts, "max": max_pts, "details": details}
        elif any(e["method"] == "curriculum" for e in log):
            acc = max((e["accuracy"] for e in log if e["method"] == "curriculum"), default=0)
            details = f"Tried curriculum but acc={acc:.2f}"
            return {"score": 2, "max": max_pts, "details": details}
        else:
            details = "Never tried curriculum learning"
            return {"score": 0, "max": max_pts, "details": details}

    def _grade_parity_invisibility(self, log, failed_set):
        """
        identified_parity_invisibility (5 pts):
        Observed that methods limited to low-order statistics fail on parity.
        Agent must have tried at least 2 methods that fail and 1 that succeeds.
        """
        max_pts = 5
        known_fail_methods = KNOWN_FAILURE_METHODS_PARITY
        observed_failures = known_fail_methods & failed_set
        has_success = len(failed_set) < len(set(e["method"] for e in log))

        if len(observed_failures) >= 2 and has_success:
            details = (
                f"Observed {len(observed_failures)} failures "
                f"({', '.join(sorted(observed_failures))}) "
                f"and found working methods -- parity structure observable"
            )
            return {"score": max_pts, "max": max_pts, "details": details}
        elif len(observed_failures) >= 1:
            details = f"Observed failure: {', '.join(sorted(observed_failures))}"
            return {"score": 2, "max": max_pts, "details": details}
        else:
            details = "Did not observe enough failures to identify parity invisibility"
            return {"score": 0, "max": max_pts, "details": details}

    def _grade_cross_challenge(self, log, solved_set, challenge):
        """
        cross_challenge_analysis (3 pts):
        Only applicable in MultiChallengeEnv. Awards points if agent solved
        methods across different challenges and could compare.
        For single-challenge episodes, award 0 with explanation.
        """
        max_pts = 3
        challenges_seen = set(e.get("challenge", challenge) for e in log)
        if len(challenges_seen) <= 1:
            details = "Single challenge episode -- cross-challenge analysis not applicable"
            return {"score": 0, "max": max_pts, "details": details}

        solved_per_challenge = {}
        for e in log:
            c = e.get("challenge", challenge)
            if e["accuracy"] >= 0.95:
                solved_per_challenge.setdefault(c, set()).add(e["method"])

        if len(solved_per_challenge) >= 2:
            details = (
                f"Solved methods across {len(solved_per_challenge)} challenges: "
                + ", ".join(f"{c}: {', '.join(sorted(ms))}" for c, ms in sorted(solved_per_challenge.items()))
            )
            return {"score": max_pts, "max": max_pts, "details": details}
        elif len(solved_per_challenge) == 1:
            details = "Only solved methods in one challenge"
            return {"score": 1, "max": max_pts, "details": details}
        else:
            details = "No methods solved across challenges"
            return {"score": 0, "max": max_pts, "details": details}

    def _grade_cache_insight(self, log, solved_set):
        """
        cache_model_insight (3 pts):
        Agent found that DMC and ARD give different rankings, implying
        cache behavior matters. Awarded if agent solved 3+ methods with
        different DMC values (showing it explored the metric space).
        """
        max_pts = 3
        dmc_values = {}
        for e in log:
            if e["accuracy"] >= 0.95 and e.get("dmc") is not None and e["dmc"] > 0:
                dmc_values[e["method"]] = e["dmc"]

        if len(dmc_values) >= 3:
            spread = max(dmc_values.values()) / min(dmc_values.values())
            details = (
                f"Measured DMC across {len(dmc_values)} methods "
                f"(spread: {spread:.0f}x) -- cache behavior observable"
            )
            return {"score": max_pts, "max": max_pts, "details": details}
        elif len(dmc_values) >= 2:
            details = f"Measured DMC for {len(dmc_values)} methods"
            return {"score": 1, "max": max_pts, "details": details}
        else:
            details = "Insufficient DMC measurements to observe cache behavior"
            return {"score": 0, "max": max_pts, "details": details}

    def _grade_algebraic(self, log, solved_set, challenge):
        """
        discovered_algebraic_solver (10 pts):
        Did the agent try GF2, KM, or SMT and get 100% accuracy?
        """
        max_pts = 10
        algebraic_solved = ALGEBRAIC_METHODS & solved_set
        if algebraic_solved:
            # Full points if any algebraic method solved it
            details = f"Solved with: {', '.join(sorted(algebraic_solved))}"
            return {"score": max_pts, "max": max_pts, "details": details}
        else:
            algebraic_tried = ALGEBRAIC_METHODS & set(e["method"] for e in log)
            if algebraic_tried:
                # Tried but didn't solve -- partial credit (3 pts for effort)
                details = (
                    f"Tried {', '.join(sorted(algebraic_tried))} "
                    f"but did not solve"
                )
                return {"score": 3, "max": max_pts, "details": details}
            else:
                details = "Never tried any algebraic method (gf2, km, smt)"
                return {"score": 0, "max": max_pts, "details": details}

    def _grade_local_learning_failure(self, log, failed_set, challenge):
        """
        identified_local_learning_failure (5 pts):
        Did the agent try methods that fail and observe low accuracy?

        The local learning methods (hebbian, predictive_coding, etc.) are not
        in the harness METHOD_MAP, so we check for forward_forward which IS
        available and is a known failure. We also check if ANY method was tried
        and failed (accuracy < 0.95) -- the agent at least observed a failure.
        """
        max_pts = 5

        # Check which failure-prone methods were tried from the harness
        # forward_forward is the one local-ish learning method in METHOD_MAP
        ff_tried = any(e["method"] == "forward_forward" for e in log)
        ff_failed = "forward_forward" in failed_set

        if ff_tried and ff_failed:
            details = (
                "Tried forward_forward and observed failure "
                "(local learning cannot detect k-th order interactions)"
            )
            return {"score": max_pts, "max": max_pts, "details": details}
        elif ff_tried:
            # Tried but somehow it solved (unlikely but possible)
            details = "Tried forward_forward but it solved (unexpected)"
            return {"score": 2, "max": max_pts, "details": details}
        else:
            # Check if agent observed any failures at all
            n_failures = len(failed_set)
            if n_failures > 0:
                details = (
                    f"Observed {n_failures} method failure(s) "
                    f"({', '.join(sorted(failed_set))}) "
                    f"but did not try forward_forward"
                )
                return {"score": 1, "max": max_pts, "details": details}
            else:
                details = "Never observed any method failure"
                return {"score": 0, "max": max_pts, "details": details}

    def _grade_metric_disagreement(self, log, solved_set):
        """
        found_metric_disagreement (5 pts):
        Did the agent try both a method that wins on ARD (KM) and one
        that wins on DMC (GF2)? Both must have solved the problem.
        """
        max_pts = 5

        solved_ard_winners = ARD_WINNERS & solved_set
        solved_dmc_winners = DMC_WINNERS & solved_set

        if solved_ard_winners and solved_dmc_winners:
            # Both KM and GF2 solved -- agent has the data to notice disagreement
            details = (
                f"Solved ARD-best ({', '.join(sorted(solved_ard_winners))}) "
                f"and DMC-best ({', '.join(sorted(solved_dmc_winners))}) -- "
                f"metric disagreement observable"
            )
            return {"score": max_pts, "max": max_pts, "details": details}
        elif solved_ard_winners or solved_dmc_winners:
            which = solved_ard_winners or solved_dmc_winners
            details = (
                f"Only solved one side: {', '.join(sorted(which))}. "
                f"Need both km and gf2 solved to observe metric disagreement."
            )
            return {"score": 2, "max": max_pts, "details": details}
        else:
            tried_either = (ARD_WINNERS | DMC_WINNERS) & set(
                e["method"] for e in log
            )
            if tried_either:
                details = (
                    f"Tried {', '.join(sorted(tried_either))} "
                    f"but none solved the problem"
                )
                return {"score": 1, "max": max_pts, "details": details}
            else:
                details = "Never tried km or gf2"
                return {"score": 0, "max": max_pts, "details": details}

    def _grade_beyond_baseline(self, log, challenge):
        """
        optimized_beyond_baseline (3 pts):
        Did the agent find a method with DMC below SGD baseline (1,278,460)?
        """
        max_pts = 3
        baseline_dmc = SGD_BASELINE_DMC

        best_dmc = float("inf")
        best_method = None
        for e in log:
            if e["accuracy"] >= 0.95 and e.get("dmc") is not None:
                if e["dmc"] < best_dmc:
                    best_dmc = e["dmc"]
                    best_method = e["method"]

        if best_method and best_dmc < baseline_dmc:
            improvement = (baseline_dmc - best_dmc) / baseline_dmc * 100
            details = (
                f"{best_method} achieved DMC {best_dmc:,.0f} "
                f"({improvement:.1f}% below SGD baseline of {baseline_dmc:,})"
            )
            return {"score": max_pts, "max": max_pts, "details": details}
        elif best_method:
            details = (
                f"Best DMC was {best_dmc:,.0f} ({best_method}), "
                f"not below SGD baseline of {baseline_dmc:,}"
            )
            return {"score": 0, "max": max_pts, "details": details}
        else:
            details = "No method solved the problem with a valid DMC"
            return {"score": 0, "max": max_pts, "details": details}

    def _grade_failure_classification(self, log, failed_set, solved_set,
                                      unique_methods):
        """
        correct_failure_classification (2 pts each):
        For each method tried that failed (accuracy < 0.95), did the agent:
          - 1pt for trying the method and observing failure
          - 2pt if it tried alternatives after the failure (moved on)
        """
        max_per_failure = 2
        max_total = 16  # from rubric

        score = 0
        failure_details = []

        for i, method in enumerate(unique_methods):
            if method not in failed_set:
                continue

            pts = 1  # Tried and observed failure

            # Check if agent tried other methods after this one failed
            # Find the last step where this method failed
            last_fail_step = None
            for e in log:
                if e["method"] == method and e["accuracy"] < 0.95:
                    last_fail_step = e["step"]

            if last_fail_step is not None:
                # Were there subsequent steps with different methods?
                subsequent = [
                    e for e in log
                    if e["step"] > last_fail_step and e["method"] != method
                ]
                if subsequent:
                    pts = 2  # Moved on to try alternatives

            score += pts
            failure_details.append(f"{method}({pts}pt)")

        score = min(score, max_total)

        if failure_details:
            details = f"Failures observed: {', '.join(failure_details)}"
        else:
            details = "No method failures observed"

        return {"score": score, "max": max_total, "details": details}

    def _grade_efficiency(self, log):
        """
        efficiency (up to 5 pts):
        How many steps to find the best method? Fewer = better.
        Max points if found in first 3 steps.
        """
        max_pts = 5

        if not log:
            return {"score": 0, "max": max_pts, "details": "No experiments run"}

        # Find the step where the overall best DMC was first achieved
        best_dmc = float("inf")
        best_step = None
        best_method = None
        for e in log:
            if e["accuracy"] >= 0.95 and e.get("dmc") is not None:
                if e["dmc"] < best_dmc:
                    best_dmc = e["dmc"]
                    best_step = e["step"]
                    best_method = e["method"]

        if best_step is None:
            return {
                "score": 0, "max": max_pts,
                "details": "Never found a method that solved the problem"
            }

        total_steps = len(log)

        # Scoring: 5 pts if found in steps 1-3, linear decay to 0 at step 16
        if best_step <= 3:
            score = max_pts
        elif best_step <= 6:
            score = 4
        elif best_step <= 9:
            score = 3
        elif best_step <= 12:
            score = 2
        elif best_step <= 15:
            score = 1
        else:
            score = 0

        details = (
            f"Best method ({best_method}, DMC={best_dmc:,.0f}) "
            f"found at step {best_step}/{total_steps}"
        )
        return {"score": score, "max": max_pts, "details": details}

    def _grade_breadth(self, log, solved_set):
        """
        exploration_breadth (up to 5 pts):
        How many distinct successful methods found? 1pt per unique method
        that achieved accuracy >= 0.95, capped at 5.
        """
        max_pts = 5
        n_solved = len(solved_set)
        score = min(n_solved, max_pts)

        if solved_set:
            details = (
                f"{n_solved} methods solved: "
                f"{', '.join(sorted(solved_set))}"
            )
        else:
            details = "No methods solved the problem"

        return {"score": score, "max": max_pts, "details": details}

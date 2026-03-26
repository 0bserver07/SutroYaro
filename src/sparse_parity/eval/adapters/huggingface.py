"""
HuggingFace Spaces leaderboard for SutroYaro eval.

A Gradio app that displays baseline results, lets users submit
a simple agent (by selecting methods in order), runs the eval,
and displays the discovery grade with per-category breakdown.

Deploy locally:
    pip install gradio
    PYTHONPATH=src python src/sparse_parity/eval/adapters/huggingface.py

Deploy to HuggingFace Spaces:
    1. Create a new Space (Gradio SDK)
    2. Copy this file + the eval/ directory
    3. Add requirements.txt: gradio, gymnasium, numpy

The ``gradio`` package is NOT required at import time. If it is
missing, importing this module still succeeds -- only create_app()
and launch will fail with a clear error message.
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# Optional import: gradio
# ---------------------------------------------------------------------------

try:
    import gradio as gr
    _HAS_GRADIO = True
except ImportError:
    gr = None  # type: ignore[assignment]
    _HAS_GRADIO = False


# ---------------------------------------------------------------------------
# Ensure eval package is importable
# ---------------------------------------------------------------------------

_EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.dirname(os.path.dirname(_EVAL_DIR))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sparse_parity.eval import registry
from sparse_parity.eval.adapters.anthropic_tools import AnthropicToolAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASELINES_PATH = os.path.join(
    _SRC_DIR, "..", "results", "eval", "baselines.json"
)
# Normalize to handle ../ components
_BASELINES_PATH = os.path.normpath(_BASELINES_PATH)


def _load_baselines():
    """Load baseline results from disk if available.

    Returns a list of dicts with keys:
        agent, mean_reward, best_method, best_score, discovery_score,
        discovery_pct, max_possible
    """
    if not os.path.exists(_BASELINES_PATH):
        return []

    with open(_BASELINES_PATH) as f:
        data = json.load(f)

    rows = []
    for r in data.get("results", []):
        grading = r.get("grading", {})
        # Most common best method across episodes
        best_methods = r.get("best_methods", [])
        if best_methods:
            from collections import Counter
            top_method = Counter(best_methods).most_common(1)[0][0] or "N/A"
        else:
            top_method = "N/A"

        # Best score (take the best non-None score)
        best_scores = [s for s in r.get("best_scores", []) if s is not None]
        best_score = min(best_scores) if best_scores else None

        rows.append({
            "agent": r.get("agent", "?"),
            "mean_reward": r.get("mean_reward", 0.0),
            "best_method": top_method,
            "best_score": best_score,
            "discovery_score": grading.get("mean_score"),
            "discovery_pct": grading.get("mean_percentage"),
            "max_possible": grading.get("max_possible"),
        })

    return rows


def _leaderboard_dataframe(baseline_rows, custom_rows=None):
    """Build a list-of-lists table for gr.Dataframe."""
    headers = [
        "Agent", "Mean Reward", "Best Method",
        "Best DMC", "Discovery Score", "Discovery %"
    ]
    rows = []
    for r in baseline_rows:
        score_str = (
            f"{r['discovery_score']:.1f}/{r['max_possible']:.0f}"
            if r.get("discovery_score") is not None else "N/A"
        )
        pct_str = (
            f"{r['discovery_pct']:.1f}%"
            if r.get("discovery_pct") is not None else "N/A"
        )
        dmc_str = (
            f"{r['best_score']:,.0f}" if r.get("best_score") is not None
            else "N/A"
        )
        rows.append([
            r["agent"],
            f"{r['mean_reward']:.4f}",
            r["best_method"],
            dmc_str,
            score_str,
            pct_str,
        ])

    if custom_rows:
        for r in custom_rows:
            score_str = (
                f"{r['discovery_score']:.1f}/{r['max_possible']:.0f}"
                if r.get("discovery_score") is not None else "N/A"
            )
            pct_str = (
                f"{r['discovery_pct']:.1f}%"
                if r.get("discovery_pct") is not None else "N/A"
            )
            dmc_str = (
                f"{r['best_score']:,.0f}" if r.get("best_score") is not None
                else "N/A"
            )
            rows.append([
                r["agent"],
                f"{r['mean_reward']:.4f}",
                r["best_method"],
                dmc_str,
                score_str,
                pct_str,
            ])

    return headers, rows


def _run_user_agent(method_sequence_str):
    """Run a user-defined agent that tries methods in the given order.

    Parameters
    ----------
    method_sequence_str : str
        Comma-separated method names, e.g. "gf2, km, sgd".

    Returns
    -------
    tuple of (result_text, grade_text, leaderboard_row)
    """
    methods = [m.strip() for m in method_sequence_str.split(",") if m.strip()]
    available = registry.list_methods()

    # Validate
    invalid = [m for m in methods if m not in available]
    if invalid:
        return (
            f"Unknown methods: {', '.join(invalid)}\n\n"
            f"Available: {', '.join(available)}",
            "",
            None,
        )

    if not methods:
        return "Please select at least one method.", "", None

    # Cap at budget
    budget = min(len(methods), 20)

    adapter = AnthropicToolAdapter(
        challenge="sparse-parity", metric="dmc", budget=budget
    )

    # Run each method
    results_lines = []
    for method in methods:
        if adapter.done:
            break
        result_json = adapter.handle_tool_call(
            "run_experiment", {"method": method}
        )
        result = json.loads(result_json)

        solved_str = "SOLVED" if result.get("solved") else "FAILED"
        dmc_str = (
            f"{result['dmc']:,.1f}" if result.get("dmc") is not None
            else "N/A"
        )
        ard_str = (
            f"{result['ard']:,.1f}" if result.get("ard") is not None
            else "N/A"
        )
        results_lines.append(
            f"Step {result.get('steps_taken', '?')}: {method} -> "
            f"{solved_str} | acc={result.get('accuracy', 0):.4f} | "
            f"DMC={dmc_str} | ARD={ard_str}"
        )

        if result.get("is_new_best"):
            results_lines.append(f"  ** New best! **")

    result_text = "\n".join(results_lines)

    # Grade
    grade = adapter.grade()
    grade_lines = [
        f"Score: {grade['total_score']}/{grade['max_possible']} "
        f"({grade['percentage']:.1f}%)",
        f"Summary: {grade['summary']}",
        "",
        "Category breakdown:",
    ]
    for cat_name, cat in grade["categories"].items():
        marker = "+" if cat["score"] > 0 else " "
        grade_lines.append(
            f"  {marker} {cat_name}: {cat['score']}/{cat['max']}"
            f"  -- {cat['details']}"
        )

    grade_text = "\n".join(grade_lines)

    # Build a leaderboard-compatible row
    # Find best method/score from the experiment log
    best_method = None
    best_score = None
    total_reward = 0.0
    for entry in adapter.experiment_log:
        total_reward += entry.get("reward", 0.0)
        if entry["accuracy"] >= 0.95 and entry.get("dmc") is not None:
            if best_score is None or entry["dmc"] < best_score:
                best_score = entry["dmc"]
                best_method = entry["method"]

    row = {
        "agent": "Your Agent",
        "mean_reward": total_reward,
        "best_method": best_method or "N/A",
        "best_score": best_score,
        "discovery_score": grade["total_score"],
        "discovery_pct": grade["percentage"],
        "max_possible": grade["max_possible"],
    }

    return result_text, grade_text, row


def _format_answer_key_summary():
    """Return a human-readable summary of the answer key / ground truth."""
    answer_key_path = os.path.join(_EVAL_DIR, "answer_key.json")
    if not os.path.exists(answer_key_path):
        return "Answer key not found."

    with open(answer_key_path) as f:
        data = json.load(f)

    rubric = data.get("grading_rubric", {})
    lines = ["Grading Rubric (7 categories, 49 points max):", ""]

    for name, info in rubric.items():
        pts = info.get("points", info.get("max_total", "?"))
        desc = info.get("description", "")
        lines.append(f"  [{pts} pts] {name}")
        lines.append(f"    {desc}")
        lines.append("")

    # Method categories from registry
    lines.append("Methods by category:")
    lines.append("")
    methods = registry.list_methods()
    by_cat = {}
    for m in methods:
        info = registry.get_method(m)
        cat = info.get("category", "other")
        by_cat.setdefault(cat, []).append(
            f"{m}: {info.get('description', '')}"
        )
    for cat, entries in sorted(by_cat.items()):
        lines.append(f"  {cat}:")
        for entry in entries:
            lines.append(f"    - {entry}")
    lines.append("")

    # Key ground truth facts
    lines.append("Key ground truth facts:")
    lines.append("  - GF(2) Gaussian elimination wins on DMC (8,607)")
    lines.append("  - KM influence estimation wins on ARD (1,585)")
    lines.append("  - SGD baseline DMC: 1,278,460")
    lines.append("  - forward_forward fails (local learning cannot detect "
                  "k-th order interactions)")
    lines.append("  - Parity is linear over GF(2) -- neural nets solve an "
                  "easy problem the hard way")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def create_app():
    """Create the Gradio Blocks app.

    Raises ImportError if gradio is not installed.
    """
    if not _HAS_GRADIO:
        raise ImportError(
            "gradio is required to run the HuggingFace Spaces app. "
            "Install it with: pip install gradio"
        )

    # Load baselines
    baseline_rows = _load_baselines()
    headers, table_data = _leaderboard_dataframe(baseline_rows)

    # Available methods for the dropdown
    methods = registry.list_methods()
    methods_by_cat = {}
    for m in methods:
        info = registry.get_method(m)
        cat = info.get("category", "other")
        methods_by_cat.setdefault(cat, []).append(m)

    method_help_lines = []
    for cat, ms in sorted(methods_by_cat.items()):
        method_help_lines.append(f"**{cat}**: {', '.join(ms)}")
    method_help = "\n\n".join(method_help_lines)

    # State for custom submissions
    custom_rows = []

    with gr.Blocks(
        title="SutroYaro Research Eval",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# SutroYaro: Can an AI Agent Do Energy-Efficient ML Research?\n\n"
            "This leaderboard evaluates how well agents explore a space of "
            "16 methods for solving sparse parity -- a benchmark task for "
            "energy-efficient AI training. Agents are scored on *discovery "
            "quality*: did they find algebraic solvers, observe metric "
            "disagreements, and classify failures correctly?"
        )

        with gr.Tab("Leaderboard"):
            gr.Markdown("### Baseline Agent Results\n"
                        "Scores from `run_eval.py` (5 episodes each, "
                        "budget=16, sparse-parity, DMC metric).")
            leaderboard = gr.Dataframe(
                headers=headers,
                value=table_data,
                interactive=False,
                wrap=True,
            )

        with gr.Tab("Try It"):
            gr.Markdown(
                "### Run Your Own Agent\n\n"
                "Enter a comma-separated list of methods to try, in order. "
                "The eval will run each method, then grade your trajectory.\n\n"
                f"{method_help}\n\n"
                "Example: `gf2, km, sgd, forward_forward, fourier`"
            )

            with gr.Row():
                method_input = gr.Textbox(
                    label="Methods (comma-separated, in order)",
                    placeholder="gf2, km, sgd, forward_forward, fourier",
                    lines=2,
                )

            run_btn = gr.Button("Run Eval", variant="primary")

            with gr.Row():
                with gr.Column():
                    result_output = gr.Textbox(
                        label="Experiment Results",
                        lines=12,
                        interactive=False,
                    )
                with gr.Column():
                    grade_output = gr.Textbox(
                        label="Discovery Grade",
                        lines=12,
                        interactive=False,
                    )

            def on_run(method_str):
                result_text, grade_text, row = _run_user_agent(method_str)

                if row is not None:
                    custom_rows.clear()
                    custom_rows.append(row)
                    _, updated_table = _leaderboard_dataframe(
                        baseline_rows, custom_rows
                    )
                else:
                    _, updated_table = _leaderboard_dataframe(baseline_rows)

                return result_text, grade_text, updated_table

            run_btn.click(
                fn=on_run,
                inputs=[method_input],
                outputs=[result_output, grade_output, leaderboard],
            )

        with gr.Tab("About"):
            gr.Markdown(
                "### What This Environment Tests\n\n"
                "The SutroYaro eval measures whether an AI agent can do "
                "*energy-efficient ML research*. The agent faces a fixed "
                "benchmark (sparse parity: learn XOR of k=3 secret bits "
                "from n=20 random inputs) and must:\n\n"
                "1. **Explore** a space of 16 methods across 4 categories "
                "(neural nets, algebraic solvers, information-theoretic, "
                "alternative)\n"
                "2. **Discover** that algebraic methods (GF2, KM, SMT) "
                "solve the problem orders of magnitude more efficiently\n"
                "3. **Observe** that DMC and ARD metrics disagree on the "
                "best method\n"
                "4. **Classify** which methods fail and why\n\n"
                "Scoring is based on *discovery quality*, not just final "
                "metric value. An agent that tries many methods and "
                "understands why some fail scores higher than one that "
                "gets lucky.\n\n"
                "### Grading Rubric\n\n"
                "| Category | Points | What it measures |\n"
                "|----------|--------|------------------|\n"
                "| discovered_algebraic_solver | 10 | Found GF2/KM/SMT |\n"
                "| identified_local_learning_failure | 5 | "
                "Tried forward_forward, saw failure |\n"
                "| found_metric_disagreement | 5 | "
                "Both KM (ARD-best) and GF2 (DMC-best) solved |\n"
                "| optimized_beyond_baseline | 3 | "
                "Beat SGD baseline DMC of 1,278,460 |\n"
                "| correct_failure_classification | 16 | "
                "2pts per correctly observed failure |\n"
                "| efficiency | 5 | "
                "Found best method in fewer steps |\n"
                "| exploration_breadth | 5 | "
                "Number of distinct methods that solved |\n"
                "| **Total** | **49** | |\n\n"
                "### Links\n\n"
                "- [GitHub repo](https://github.com/cybertronai/SutroYaro)\n"
                "- [Eval README](https://github.com/cybertronai/SutroYaro/"
                "tree/main/src/sparse_parity/eval)\n"
                "- [Sutro Group](https://cybertronai.github.io/SutroYaro/)\n"
            )

        with gr.Tab("Answer Key"):
            gr.Markdown("### Ground Truth Summary\n\n"
                        "What the eval is looking for and why.")
            answer_key_text = gr.Textbox(
                value=_format_answer_key_summary(),
                label="Answer Key / Ground Truth",
                lines=30,
                interactive=False,
            )

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Launch the Gradio app locally."""
    if not _HAS_GRADIO:
        print(
            "Error: gradio is not installed.\n"
            "Install it with: pip install gradio\n"
            "Then re-run: python src/sparse_parity/eval/adapters/huggingface.py"
        )
        sys.exit(1)

    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()

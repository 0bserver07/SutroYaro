"""
Inspect AI task for SutroYaro eval.

Prototype adapter for the UK AISI Inspect framework. Wraps the
SutroYaro environment as an Inspect task so it can be evaluated with:

    inspect eval src/sparse_parity/eval/adapters/inspect_task.py

Requires: pip install inspect-ai

This is a structural prototype. The Inspect framework uses a different
execution model from raw tool-use -- it manages the conversation loop
internally. This adapter maps the SutroYaro environment into Inspect's
Task / Solver / Scorer abstractions.

Architecture:
    Task        = one episode of the SutroYaro env
    Solver      = the LLM agent (Inspect handles this)
    Scorer      = DiscoveryGrader from grader.py
    Tools       = run_experiment, check_status, read_experiment_log
    Dataset     = one sample per challenge (sparse-parity, sparse-sum, sparse-and)
"""

import json


# ---------------------------------------------------------------------------
# Shared state: one adapter instance per task invocation.
# Inspect tools are plain functions, so we use module-level state.
# ---------------------------------------------------------------------------

_adapter_instance = None


def _get_adapter():
    """Return the current adapter instance (set by the solver)."""
    global _adapter_instance
    if _adapter_instance is None:
        raise RuntimeError(
            "Adapter not initialized. The solver must call _init_adapter() first."
        )
    return _adapter_instance


def _init_adapter(challenge="sparse-parity", metric="dmc", budget=20, **kwargs):
    """Initialize the adapter for a new episode."""
    global _adapter_instance
    # Lazy import so the eval package is only loaded when actually needed
    from sparse_parity.eval.adapters.anthropic_tools import AnthropicToolAdapter
    _adapter_instance = AnthropicToolAdapter(
        challenge=challenge, metric=metric, budget=budget, **kwargs
    )
    return _adapter_instance


# ---------------------------------------------------------------------------
# Task creation
# ---------------------------------------------------------------------------

def create_inspect_task(
    challenge="sparse-parity",
    metric="dmc",
    budget=20,
):
    """Create an Inspect-compatible task definition.

    Parameters
    ----------
    challenge : str
        Challenge name (default "sparse-parity").
    metric : str
        Target metric (default "dmc").
    budget : int
        Experiment budget (default 20).

    Returns
    -------
    inspect_ai.Task or None
        The task object, or None if inspect-ai is not installed.
    """
    try:
        from inspect_ai import Task, task
        from inspect_ai.dataset import Sample, MemoryDataset
        from inspect_ai.scorer import Scorer, Score, Target, scorer
        from inspect_ai.solver import (
            Solver,
            TaskState,
            generate,
            solver,
            use_tools,
        )
        from inspect_ai.tool import Tool, tool
    except ImportError:
        print(
            "Inspect AI not installed. Run: pip install inspect-ai\n"
            "Then: inspect eval src/sparse_parity/eval/adapters/inspect_task.py"
        )
        return None

    # ---- Tools (Inspect @tool functions) ----

    @tool
    def run_experiment():
        """Run a method on the current challenge and observe energy metrics."""

        async def execute(method: str) -> str:
            """Run the specified method.

            Args:
                method: Which method to try (e.g. "gf2", "sgd", "km").
            """
            adapter = _get_adapter()
            return adapter.handle_tool_call("run_experiment", {"method": method})

        return execute

    @tool
    def check_status():
        """Check current best score, budget remaining, and methods tried so far."""

        async def execute() -> str:
            """Check the current evaluation status."""
            adapter = _get_adapter()
            return adapter.handle_tool_call("check_status", {})

        return execute

    @tool
    def read_experiment_log():
        """Read the full log of experiments run so far with their results."""

        async def execute() -> str:
            """Read the experiment log."""
            adapter = _get_adapter()
            return adapter.handle_tool_call("read_experiment_log", {})

        return execute

    # ---- Solver (init adapter, then let the model use tools) ----

    @solver
    def sutro_solver():
        """Solver that initializes the env and lets the model run experiments."""

        async def solve(state: TaskState, generate_fn) -> TaskState:
            # Initialize the adapter for this sample's challenge
            sample_challenge = state.metadata.get("challenge", challenge)
            sample_metric = state.metadata.get("metric", metric)
            sample_budget = state.metadata.get("budget", budget)
            _init_adapter(
                challenge=sample_challenge,
                metric=sample_metric,
                budget=sample_budget,
            )

            # Let the model generate with tools available
            tool_solver = use_tools([
                run_experiment(),
                check_status(),
                read_experiment_log(),
            ])
            state = await tool_solver(state, generate_fn)

            # Run generate in a loop until the model stops calling tools
            # or budget is exhausted
            max_iters = budget + 10  # some slack for check_status calls
            for _ in range(max_iters):
                state = await generate_fn(state)
                if not state.output or not any(
                    getattr(block, "type", None) == "tool_use"
                    for block in (state.output.choices[0].message.content
                                  if state.output.choices else [])
                ):
                    break

            return state

        return solve

    # ---- Scorer (uses DiscoveryGrader) ----

    @scorer(metrics=[])
    def sutro_scorer():
        """Score based on the DiscoveryGrader report."""

        async def score(state: TaskState, target: Target) -> Score:
            adapter = _get_adapter()
            if adapter is None:
                return Score(value=0.0, explanation="Adapter not initialized")

            grade = adapter.grade()
            return Score(
                value=grade["percentage"] / 100.0,  # normalize to [0, 1]
                answer=grade["summary"],
                explanation=grade["full_report"],
                metadata={
                    "total_score": grade["total_score"],
                    "max_possible": grade["max_possible"],
                    "categories": grade["categories"],
                },
            )

        return score

    # ---- Dataset (one sample per challenge) ----

    # Get available challenges from registry
    try:
        from sparse_parity.eval import registry
        challenges = registry.list_challenges()
    except Exception:
        challenges = ["sparse-parity", "sparse-sum", "sparse-and"]

    adapter_for_prompt = _init_adapter(challenge=challenge, metric=metric, budget=budget)
    system_prompt = adapter_for_prompt.get_system_prompt()

    samples = []
    for ch in challenges:
        samples.append(Sample(
            input=system_prompt + (
                "\n\nYou have a budget of experiments to find the most "
                "energy-efficient method. Start by exploring, then narrow down. "
                "Think step by step about which methods to try and why."
            ),
            target=f"Find best {metric} method for {ch}",
            metadata={
                "challenge": ch,
                "metric": metric,
                "budget": budget,
            },
        ))

    dataset = MemoryDataset(samples=samples, name="sutro-challenges")

    # ---- Task ----

    return Task(
        dataset=dataset,
        solver=sutro_solver(),
        scorer=sutro_scorer(),
    )


# ---------------------------------------------------------------------------
# Inspect entry point: ``inspect eval`` looks for a @task function
# ---------------------------------------------------------------------------

def _try_create_task():
    """Entry point for ``inspect eval``. Returns Task or prints install instructions."""
    task = create_inspect_task()
    if task is None:
        raise SystemExit(
            "Cannot create task: inspect-ai is not installed.\n"
            "Run: pip install inspect-ai"
        )
    return task


# If inspect-ai is available, register the task with the @task decorator
try:
    from inspect_ai import task as _task_decorator

    @_task_decorator
    def sutro_yaro():
        """SutroYaro energy-efficient method selection task."""
        return create_inspect_task()

except ImportError:
    # inspect-ai not installed -- provide a helpful stub
    def sutro_yaro():
        """SutroYaro task (requires inspect-ai). Run: pip install inspect-ai"""
        print("Inspect AI not installed. Run: pip install inspect-ai")
        return None

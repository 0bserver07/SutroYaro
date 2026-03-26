"""
Anthropic tool-use adapter for SutroYaro eval environment.

Wraps SutroYaroEnv as a set of Claude tool-use calls so an LLM agent
can interact with the environment through the Anthropic Messages API
instead of discrete action indices.

The ``anthropic`` package is NOT required at import time -- it is only
needed when you actually send messages to the API. This module only
builds the tool definitions and translates tool calls into env actions.

Usage:

    from sparse_parity.eval.adapters.anthropic_tools import AnthropicToolAdapter

    adapter = AnthropicToolAdapter(challenge="sparse-parity", metric="dmc", budget=20)

    # Get tool definitions for the Anthropic API
    tools = adapter.get_tools()

    # Get a system prompt explaining the task
    system = adapter.get_system_prompt()

    # In your message loop, handle tool calls:
    result_text = adapter.handle_tool_call("run_experiment", {"method": "gf2"})

    # After the episode, grade the trajectory:
    report = adapter.grade()
"""

import json

from sparse_parity.eval import registry
from sparse_parity.eval.env import SutroYaroEnv
from sparse_parity.eval.grader import DiscoveryGrader


def _build_tools():
    """Build the tool definitions from the current registry state.

    Returns a list of dicts in Anthropic tool-use schema format.
    """
    methods = registry.list_methods()
    method_descriptions = []
    for m in methods:
        info = registry.get_method(m)
        desc = info.get("description", "")
        cat = info.get("category", "")
        method_descriptions.append(f"{m} ({cat}): {desc}")

    method_help = "\n".join(f"  - {d}" for d in method_descriptions)

    return [
        {
            "name": "run_experiment",
            "description": (
                "Run a method on the current challenge and observe energy metrics. "
                "Returns accuracy, ARD (average reuse distance), DMC (data movement "
                "complexity), wall-clock time, and whether the method solved the task.\n\n"
                "Available methods:\n" + method_help
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": methods,
                        "description": "Which method to run.",
                    }
                },
                "required": ["method"],
            },
        },
        {
            "name": "check_status",
            "description": (
                "Check current best score, budget remaining, methods tried so far, "
                "and the target metric. Use this to plan your next experiment."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "read_experiment_log",
            "description": (
                "Read the full log of experiments run so far with their results. "
                "Each entry shows the method, accuracy, ARD, DMC, time, and whether "
                "it set a new best score."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
    ]


class AnthropicToolAdapter:
    """Adapts SutroYaroEnv for use with Anthropic's tool-use API.

    Translates between the Gymnasium discrete-action interface and
    Claude tool calls. The LLM sees three tools (run_experiment,
    check_status, read_experiment_log) instead of raw action indices.

    Parameters
    ----------
    challenge : str
        Which challenge to run (default "sparse-parity").
    metric : str
        Target metric: "dmc" or "ard" (default "dmc").
    budget : int
        Maximum number of experiments the agent can run (default 20).
    **env_kwargs
        Extra keyword arguments forwarded to SutroYaroEnv (e.g. n_bits,
        k_sparse, seed, backend).
    """

    def __init__(self, challenge="sparse-parity", metric="dmc", budget=20,
                 **env_kwargs):
        self.env = SutroYaroEnv(
            challenge=challenge, metric=metric, budget=budget, **env_kwargs
        )
        self._obs, self._info = self.env.reset()
        self._terminated = False
        self._truncated = False
        self._grader = DiscoveryGrader()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tools(self):
        """Return tool definitions for the Anthropic Messages API.

        These go into the ``tools`` parameter of ``client.messages.create()``.
        """
        return _build_tools()

    def handle_tool_call(self, tool_name, tool_input):
        """Execute a tool call and return the result as a JSON string.

        Parameters
        ----------
        tool_name : str
            One of "run_experiment", "check_status", "read_experiment_log".
        tool_input : dict
            The input object from the tool call. For run_experiment this
            must contain {"method": "<method_name>"}.

        Returns
        -------
        str
            JSON-formatted result string suitable for a tool_result content block.

        Raises
        ------
        ValueError
            If tool_name is unknown or if the episode has ended.
        """
        if tool_name == "run_experiment":
            return self._handle_run_experiment(tool_input)
        elif tool_name == "check_status":
            return self._handle_check_status()
        elif tool_name == "read_experiment_log":
            return self._handle_read_log()
        else:
            return json.dumps({
                "error": f"Unknown tool: {tool_name}. "
                         f"Available tools: run_experiment, check_status, read_experiment_log"
            })

    def get_system_prompt(self):
        """Return a system prompt explaining the task to the LLM.

        Includes the challenge description, metric goal, budget, and
        available methods with their categories.
        """
        challenge = self.env.challenge
        metric = self.env.metric
        budget = self.env.budget
        n_bits = self.env.n_bits
        k_sparse = self.env.k_sparse

        # Get challenge description from registry
        challenge_info = registry.get_challenge(challenge)
        challenge_desc = challenge_info.get("description", challenge)

        # Build method list with categories
        methods = registry.list_methods()
        by_category = {}
        for m in methods:
            info = registry.get_method(m)
            cat = info.get("category", "other")
            by_category.setdefault(cat, []).append(
                f"{m}: {info.get('description', '')}"
            )

        method_section = ""
        for cat, entries in by_category.items():
            method_section += f"\n  {cat}:\n"
            for entry in entries:
                method_section += f"    - {entry}\n"

        metric_name = "Data Movement Complexity (DMC)" if metric == "dmc" else "Average Reuse Distance (ARD)"

        return (
            f"You are an AI researcher running experiments to find the most "
            f"energy-efficient method for a sparse learning task.\n\n"
            f"CHALLENGE: {challenge}\n"
            f"{challenge_desc}\n"
            f"Configuration: n_bits={n_bits}, k_sparse={k_sparse}\n\n"
            f"GOAL: Find the method with the lowest {metric_name} ({metric.upper()}).\n"
            f"A method must achieve accuracy >= 0.95 to count as solving the task.\n\n"
            f"BUDGET: You can run at most {budget} experiments. Use them wisely.\n"
            f"Think about which methods are likely to work and why. Consider trying\n"
            f"diverse approaches (neural nets, algebraic solvers, information-theoretic\n"
            f"methods) to understand the problem structure.\n\n"
            f"AVAILABLE METHODS:{method_section}\n"
            f"TOOLS:\n"
            f"  - run_experiment: Run a method and see its energy metrics.\n"
            f"  - check_status: See your current best score and remaining budget.\n"
            f"  - read_experiment_log: Review all experiments you have run so far.\n\n"
            f"After each experiment, reason about what the result tells you about\n"
            f"the problem. Which methods exploit the structure of the task? Which\n"
            f"categories of methods fail, and why?"
        )

    def grade(self):
        """Grade the current trajectory using DiscoveryGrader.

        Returns
        -------
        dict
            Grading report with keys: total_score, max_possible, percentage,
            categories, summary.
        """
        report = self._grader.grade_episode(self.env)
        return {
            "total_score": report.total_score,
            "max_possible": report.max_possible,
            "percentage": report.percentage,
            "categories": report.categories,
            "summary": report.summary,
            "full_report": str(report),
        }

    @property
    def done(self):
        """True if the episode has ended (budget exhausted or terminated)."""
        return self._terminated or self._truncated

    @property
    def experiment_log(self):
        """The raw experiment log from the underlying environment."""
        return self.env.experiment_log

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_run_experiment(self, tool_input):
        """Run one experiment through the environment."""
        if self.done:
            return json.dumps({
                "error": "Episode has ended. No more experiments can be run.",
                "budget_remaining": 0,
            })

        method = tool_input.get("method")
        if method is None:
            return json.dumps({
                "error": "Missing required parameter: method",
            })

        # Translate method name to action index
        methods = registry.list_methods()
        if method not in methods:
            return json.dumps({
                "error": f"Unknown method: {method}. "
                         f"Available methods: {methods}",
            })

        action = registry.get_method_index(method)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs = obs
        self._terminated = terminated
        self._truncated = truncated

        # Build a human-readable result
        accuracy = info.get("accuracy", 0.0)
        solved = accuracy >= 0.95
        ard = info.get("ard")
        dmc = info.get("dmc")
        time_s = info.get("time_s")
        error = info.get("error")

        result = {
            "method": method,
            "solved": solved,
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
            "ard": round(ard, 1) if ard is not None else None,
            "dmc": round(dmc, 1) if dmc is not None else None,
            "time_s": round(time_s, 4) if time_s is not None else None,
            "is_new_best": info.get("is_new_best", False),
            "budget_remaining": max(0, self.env.budget - self.env.steps_taken),
            "steps_taken": self.env.steps_taken,
        }

        if error:
            result["error"] = error

        if info.get("is_new_best"):
            result["message"] = (
                f"New best! {method} achieved {self.env.metric.upper()} = "
                f"{dmc if self.env.metric == 'dmc' else ard}"
            )

        if self._truncated:
            result["message"] = result.get("message", "") + " Budget exhausted."

        return json.dumps(result, default=str)

    def _handle_check_status(self):
        """Return current status summary."""
        methods = registry.list_methods()
        tried = [
            methods[i]
            for i in range(len(methods))
            if i < len(self.env.methods_tried) and self.env.methods_tried[i]
        ]

        best_score = self.env.best_score
        if best_score == float("inf"):
            best_score_str = "No method has solved the task yet"
        else:
            best_score_str = f"{best_score:,.1f}"

        status = {
            "challenge": self.env.challenge,
            "metric": self.env.metric,
            "best_score": best_score_str,
            "budget_remaining": max(0, self.env.budget - self.env.steps_taken),
            "budget_total": self.env.budget,
            "steps_taken": self.env.steps_taken,
            "methods_tried": tried,
            "methods_not_tried": [m for m in methods if m not in tried],
            "episode_done": self.done,
        }
        return json.dumps(status, default=str)

    def _handle_read_log(self):
        """Return the full experiment log."""
        log = self.env.experiment_log
        if not log:
            return json.dumps({"message": "No experiments run yet.", "log": []})

        # Clean up the log for readability
        clean_log = []
        for entry in log:
            clean_entry = {
                "step": entry["step"],
                "method": entry["method"],
                "accuracy": round(entry["accuracy"], 4) if entry["accuracy"] is not None else None,
                "ard": round(entry["ard"], 1) if entry.get("ard") is not None else None,
                "dmc": round(entry["dmc"], 1) if entry.get("dmc") is not None else None,
                "time_s": round(entry["time_s"], 4) if entry.get("time_s") is not None else None,
                "solved": entry["accuracy"] >= 0.95 if entry["accuracy"] is not None else False,
                "is_new_best": entry.get("is_new_best", False),
            }
            if entry.get("error"):
                clean_entry["error"] = entry["error"]
            clean_log.append(clean_entry)

        return json.dumps({"log": clean_log}, default=str)


# ------------------------------------------------------------------
# Convenience: run a full eval loop with the Anthropic API
# ------------------------------------------------------------------

def run_anthropic_eval(
    model="claude-sonnet-4-20250514",
    challenge="sparse-parity",
    metric="dmc",
    budget=20,
    max_turns=50,
    verbose=True,
    **env_kwargs,
):
    """Run a full evaluation episode using the Anthropic Messages API.

    Requires the ``anthropic`` package to be installed.

    Parameters
    ----------
    model : str
        Which Claude model to use.
    challenge : str
        Challenge name (default "sparse-parity").
    metric : str
        Target metric (default "dmc").
    budget : int
        Experiment budget (default 20).
    max_turns : int
        Maximum conversation turns to prevent runaway loops (default 50).
    verbose : bool
        Print each tool call and result (default True).
    **env_kwargs
        Extra args forwarded to SutroYaroEnv.

    Returns
    -------
    dict
        {"grade": <grade_dict>, "messages": <message_list>, "turns": <int>}
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The anthropic package is required to run the eval loop. "
            "Install it with: pip install anthropic"
        )

    adapter = AnthropicToolAdapter(
        challenge=challenge, metric=metric, budget=budget, **env_kwargs
    )
    client = anthropic.Anthropic()

    system = adapter.get_system_prompt()
    tools = adapter.get_tools()
    messages = []

    # Initial user message to kick things off
    messages.append({
        "role": "user",
        "content": (
            "You have a budget of experiments to find the most energy-efficient "
            "method. Start by exploring, then narrow down. Think step by step "
            "about which methods to try and why."
        ),
    })

    turns = 0
    while turns < max_turns and not adapter.done:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        # Check if there are tool calls to handle
        tool_results = []
        has_tool_use = False
        for block in response.content:
            if block.type == "tool_use":
                has_tool_use = True
                if verbose:
                    print(f"[Turn {turns+1}] Tool: {block.name}({block.input})")

                result_text = adapter.handle_tool_call(block.name, block.input)

                if verbose:
                    print(f"  Result: {result_text[:200]}...")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        if not has_tool_use:
            # Model finished reasoning without calling tools -- done
            if verbose:
                for block in response.content:
                    if hasattr(block, "text"):
                        print(f"[Turn {turns+1}] Text: {block.text[:300]}")
            break

        # Feed tool results back
        messages.append({"role": "user", "content": tool_results})
        turns += 1

    # Grade the trajectory
    grade = adapter.grade()
    if verbose:
        print(f"\n{'='*60}")
        print(grade["full_report"])

    return {
        "grade": grade,
        "messages": messages,
        "turns": turns,
    }

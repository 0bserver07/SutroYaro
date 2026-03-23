"""
PrimeIntellect Environment Hub adapter for SutroYaro eval.

Wraps the SutroYaro evaluation environment as a verifiers-compatible
environment for PrimeIntellect's community environment system.

Setup:
    pip install prime-python verifiers
    prime env init sutro-parity
    # Copy this file's load_environment into the generated template
    prime env push

See: https://github.com/PrimeIntellect-ai/verifiers
See: https://github.com/PrimeIntellect-ai/community-environments

The ``verifiers`` package is NOT required at import time. The adapter
works standalone (test_without_verifiers) for local testing. When
verifiers is installed, load_environment() returns a vf.Environment
ready for PrimeIntellect's platform.

Architecture:
    Dataset     = one row per challenge configuration (parity, sum, and)
    Rubric      = score_trajectory wrapping DiscoveryGrader
    Tools       = reused from AnthropicToolAdapter (run_experiment, check_status, read_experiment_log)
    Environment = vf.SingleTurnEnv (dataset + rubric)
"""

import json

from sparse_parity.eval import registry
from sparse_parity.eval.adapters.anthropic_tools import AnthropicToolAdapter


# ---------------------------------------------------------------------------
# Dataset: one row per challenge configuration
# ---------------------------------------------------------------------------

CHALLENGES = ["sparse-parity", "sparse-sum", "sparse-and"]

DATASET = [
    {
        "challenge": ch,
        "n_bits": 20,
        "k_sparse": 3,
        "metric": "dmc",
        "budget": 20,
    }
    for ch in CHALLENGES
]


# ---------------------------------------------------------------------------
# Tool definitions (reused from the Anthropic adapter)
# ---------------------------------------------------------------------------

def get_tool_definitions():
    """Return tool definitions in a platform-neutral format.

    These are the same tools the LLM agent gets in the Anthropic adapter,
    extracted here so PrimeIntellect's system prompt / tool registration
    can reuse them.
    """
    # Build a temporary adapter to get the tool defs
    adapter = AnthropicToolAdapter(challenge="sparse-parity", metric="dmc", budget=20)
    return adapter.get_tools()


def get_system_prompt(challenge="sparse-parity", metric="dmc", budget=20):
    """Return the system prompt for a given challenge configuration."""
    adapter = AnthropicToolAdapter(challenge=challenge, metric=metric, budget=budget)
    return adapter.get_system_prompt()


# ---------------------------------------------------------------------------
# Rubric: maps DiscoveryGrader score to 0.0-1.0
# ---------------------------------------------------------------------------

def _parse_tool_calls(completion_text):
    """Extract tool calls from a completion string.

    PrimeIntellect completions may contain tool calls in various formats.
    This parser handles the common patterns:
      - JSON blocks with {"tool": "name", "input": {...}}
      - function_call style: run_experiment(method="gf2")

    Returns a list of (tool_name, tool_input) tuples.
    """
    calls = []

    # Try to find JSON tool call blocks
    # Look for patterns like {"tool": "run_experiment", "input": {"method": "gf2"}}
    import re

    # Pattern 1: JSON objects with "tool" and "input" keys
    json_pattern = re.compile(
        r'\{[^{}]*"tool"\s*:\s*"(\w+)"[^{}]*"input"\s*:\s*(\{[^{}]*\})[^{}]*\}',
        re.DOTALL,
    )
    for match in json_pattern.finditer(completion_text):
        tool_name = match.group(1)
        try:
            tool_input = json.loads(match.group(2))
        except json.JSONDecodeError:
            tool_input = {}
        calls.append((tool_name, tool_input))

    # Pattern 2: function-call style: run_experiment(method="gf2")
    if not calls:
        func_pattern = re.compile(
            r'(run_experiment|check_status|read_experiment_log)'
            r'\s*\(\s*(.*?)\s*\)',
            re.DOTALL,
        )
        for match in func_pattern.finditer(completion_text):
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            tool_input = {}
            if args_str and tool_name == "run_experiment":
                # Parse method="gf2" or method='gf2'
                method_match = re.search(
                    r'method\s*=\s*["\'](\w+)["\']', args_str
                )
                if method_match:
                    tool_input["method"] = method_match.group(1)
            calls.append((tool_name, tool_input))

    return calls


async def score_trajectory(completion, answer) -> float:
    """Score an agent's trajectory from its completion text.

    Replays the tool calls found in the completion through an
    AnthropicToolAdapter, then runs DiscoveryGrader to produce
    a normalized score in [0.0, 1.0].

    Parameters
    ----------
    completion : str
        The full text of the agent's completion (may contain tool
        calls in JSON or function-call format).
    answer : dict
        The dataset row for this challenge (from DATASET).

    Returns
    -------
    float
        Normalized score between 0.0 and 1.0.
    """
    # Parse challenge config from the answer/dataset row
    if isinstance(answer, dict):
        challenge = answer.get("challenge", "sparse-parity")
        metric = answer.get("metric", "dmc")
        budget = answer.get("budget", 20)
    else:
        challenge = "sparse-parity"
        metric = "dmc"
        budget = 20

    # Create a fresh adapter for this episode
    adapter = AnthropicToolAdapter(
        challenge=challenge, metric=metric, budget=budget
    )

    # Parse and replay tool calls from the completion
    calls = _parse_tool_calls(completion if isinstance(completion, str) else str(completion))

    for tool_name, tool_input in calls:
        if adapter.done:
            break
        try:
            adapter.handle_tool_call(tool_name, tool_input)
        except Exception:
            # Skip malformed calls
            continue

    # Grade the trajectory
    grade = adapter.grade()
    normalized = grade["percentage"] / 100.0
    return max(0.0, min(1.0, normalized))


def score_trajectory_sync(completion, answer) -> float:
    """Synchronous version of score_trajectory for local testing."""
    import asyncio
    return asyncio.run(score_trajectory(completion, answer))


# ---------------------------------------------------------------------------
# Environment loader (PrimeIntellect entry point)
# ---------------------------------------------------------------------------

def load_environment(challenge="sparse-parity"):
    """Load SutroYaro as a PrimeIntellect verifiers environment.

    This is the entry point that PrimeIntellect's platform calls.
    It returns a vf.Environment with dataset and rubric configured.

    Parameters
    ----------
    challenge : str
        Which challenge to load. Use "all" for the full dataset
        (sparse-parity, sparse-sum, sparse-and). Default is
        "sparse-parity".

    Returns
    -------
    vf.Environment or None
        The environment object, or None if verifiers is not installed.
    """
    try:
        import verifiers as vf
    except ImportError:
        print("=" * 60)
        print("PrimeIntellect verifiers library not installed.")
        print()
        print("To install:")
        print("  pip install prime-python verifiers")
        print()
        print("To set up as a PrimeIntellect community environment:")
        print("  prime env init sutro-parity")
        print("  # Copy load_environment() into the generated template")
        print("  prime env push")
        print()
        print("See: https://github.com/PrimeIntellect-ai/verifiers")
        print("See: https://github.com/PrimeIntellect-ai/community-environments")
        print("=" * 60)
        return None

    # Build dataset rows
    if challenge == "all":
        rows = DATASET
    else:
        rows = [row for row in DATASET if row["challenge"] == challenge]
        if not rows:
            # Fall back to default config for unknown challenge
            rows = [
                {
                    "challenge": challenge,
                    "n_bits": 20,
                    "k_sparse": 3,
                    "metric": "dmc",
                    "budget": 20,
                }
            ]

    # Build the verifiers dataset
    dataset = vf.Dataset(rows)

    # Build the rubric with our scoring function
    rubric = vf.Rubric(funcs=[score_trajectory])

    # Build the system prompt (used as task instructions)
    system_prompt = get_system_prompt(
        challenge=rows[0]["challenge"],
        metric=rows[0]["metric"],
        budget=rows[0]["budget"],
    )

    # Build tool definitions for the agent
    tools = get_tool_definitions()

    # Create the environment
    env = vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        tools=tools,
        system_prompt=system_prompt,
    )

    return env


# ---------------------------------------------------------------------------
# Standalone test (works without verifiers installed)
# ---------------------------------------------------------------------------

def test_without_verifiers():
    """Run the eval locally to verify scoring logic works.

    This exercises the full pipeline: adapter creation, tool calls,
    grading, and score normalization -- all without needing the
    verifiers library.
    """
    print("Testing PrimeIntellect adapter (standalone mode)")
    print("=" * 60)

    # Test each challenge
    for row in DATASET:
        challenge = row["challenge"]
        print(f"\nChallenge: {challenge}")
        print("-" * 40)

        adapter = AnthropicToolAdapter(
            challenge=challenge,
            metric=row["metric"],
            budget=row["budget"],
        )

        # Simulate an agent trying several methods
        methods_to_try = ["sgd", "gf2", "km", "forward_forward"]
        for method in methods_to_try:
            if adapter.done:
                break
            try:
                result = adapter.handle_tool_call(
                    "run_experiment", {"method": method}
                )
                parsed = json.loads(result)
                solved = parsed.get("solved", False)
                dmc = parsed.get("dmc", "N/A")
                acc = parsed.get("accuracy", "N/A")
                print(f"  {method:20s}  solved={solved}  acc={acc}  dmc={dmc}")
            except Exception as e:
                print(f"  {method:20s}  error: {e}")

        # Grade the trajectory
        grade = adapter.grade()
        normalized = grade["percentage"] / 100.0
        print(f"\n  Score: {grade['total_score']}/{grade['max_possible']} = {normalized:.2f}")
        print(f"  Summary: {grade['summary']}")

    # Also test the completion parser
    print("\n" + "=" * 60)
    print("Testing completion parser")
    print("-" * 40)

    test_completion = '''
    I'll try GF2 first since it's an algebraic solver.
    {"tool": "run_experiment", "input": {"method": "gf2"}}
    That worked well. Let me also try KM.
    {"tool": "run_experiment", "input": {"method": "km"}}
    And check status.
    {"tool": "check_status", "input": {}}
    '''

    calls = _parse_tool_calls(test_completion)
    print(f"  Parsed {len(calls)} tool calls from completion text:")
    for name, inp in calls:
        print(f"    {name}({inp})")

    # Score the test completion
    test_answer = DATASET[0]  # sparse-parity
    score = score_trajectory_sync(test_completion, test_answer)
    print(f"  Trajectory score: {score:.2f}")

    print("\n" + "=" * 60)
    print("Standalone test complete.")
    print(
        "To use with PrimeIntellect, install verifiers and call "
        "load_environment()."
    )

    return normalized


if __name__ == "__main__":
    test_without_verifiers()

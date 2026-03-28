# Eval Platform Adapters

Wrappers that adapt the SutroYaro Gymnasium environment for different
LLM evaluation platforms. The core environment (`env.py`) exposes a
discrete action space (integer indices); these adapters translate that
into tool-use calls, Inspect tasks, or other platform-specific formats.

## Adapters

### `anthropic_tools.py` -- Anthropic tool-use adapter

Wraps the environment as three Claude tools:

| Tool | Description |
|------|-------------|
| `run_experiment` | Run a method and observe energy metrics |
| `check_status` | Check best score, budget remaining, methods tried |
| `read_experiment_log` | Read the full experiment history |

The adapter does NOT import the `anthropic` package at module load time.
You only need it installed when calling `run_anthropic_eval()`.

**Quick start (manual loop):**

```python
from sparse_parity.eval.adapters.anthropic_tools import AnthropicToolAdapter

adapter = AnthropicToolAdapter(
    challenge="sparse-parity", metric="dmc", budget=20
)

# Get tool definitions and system prompt
tools = adapter.get_tools()      # list of dicts for Anthropic API
system = adapter.get_system_prompt()  # str

# In your message loop, handle tool calls:
result_json = adapter.handle_tool_call("run_experiment", {"method": "gf2"})
result_json = adapter.handle_tool_call("check_status", {})

# After the episode:
report = adapter.grade()
print(report["full_report"])
```

**Quick start (automatic eval loop):**

```python
from sparse_parity.eval.adapters.anthropic_tools import run_anthropic_eval

# Requires ANTHROPIC_API_KEY env var
result = run_anthropic_eval(
    model="claude-sonnet-4-20250514",
    challenge="sparse-parity",
    metric="dmc",
    budget=20,
    verbose=True,
)

print(f"Score: {result['grade']['percentage']:.0f}%")
print(f"Turns: {result['turns']}")
```

### `inspect_task.py` -- UK AISI Inspect framework (prototype)

Structural prototype for the [Inspect](https://inspect.ai-safety-institute.org.uk/)
evaluation framework.

```bash
# Install
pip install inspect-ai

# Run eval
inspect eval src/sparse_parity/eval/adapters/inspect_task.py
```

Or programmatically:

```python
from sparse_parity.eval.adapters.inspect_task import create_inspect_task

task = create_inspect_task(challenge="sparse-parity", metric="dmc", budget=20)
# Pass to inspect_ai.eval() or inspect CLI
```

The Inspect adapter reuses the same `AnthropicToolAdapter` internally,
mapping it into Inspect's Task/Solver/Scorer abstractions:

- **Task**: one episode of the SutroYaro env
- **Solver**: initializes the env, then hands control to the LLM with tools
- **Scorer**: runs `DiscoveryGrader` and returns a normalized score in [0, 1]
- **Dataset**: one sample per challenge (sparse-parity, sparse-sum, sparse-and)

### `primeintellect.py` -- PrimeIntellect Environment Hub adapter

Wraps the eval as a [verifiers](https://github.com/PrimeIntellect-ai/verifiers)-compatible
environment for PrimeIntellect's community environment system.

```bash
# Install
pip install prime-python verifiers

# Set up as a community environment
prime env init sutro-parity
# Copy load_environment() into the generated template
prime env push
```

Or load programmatically:

```python
from sparse_parity.eval.adapters.primeintellect import load_environment

env = load_environment(challenge="sparse-parity")
# Returns a vf.SingleTurnEnv with dataset and rubric
```

The adapter does NOT require the `verifiers` package at import time. You
can test the full scoring pipeline locally without it:

```python
from sparse_parity.eval.adapters.primeintellect import test_without_verifiers

test_without_verifiers()
```

Or run it directly:

```bash
PYTHONPATH=src python -m sparse_parity.eval.adapters.primeintellect
```

Key components:

- **Dataset**: one row per challenge (sparse-parity, sparse-sum, sparse-and)
- **Rubric**: `score_trajectory` parses tool calls from the completion, replays them through `AnthropicToolAdapter`, and runs `DiscoveryGrader` to produce a [0, 1] score
- **Tools**: reused from `anthropic_tools.py` (run_experiment, check_status, read_experiment_log)
- **Completion parser**: extracts tool calls from JSON blocks or function-call syntax in the agent's output

### `huggingface.py` -- HuggingFace Spaces Gradio leaderboard

Interactive web app that displays baseline results, lets users submit
a simple agent (by selecting methods in order), runs the eval, and
shows discovery grades with per-category breakdown.

```bash
# Install
pip install gradio

# Run locally
PYTHONPATH=src python src/sparse_parity/eval/adapters/huggingface.py
```

Or deploy to HuggingFace Spaces:

1. Create a new Space (select Gradio SDK)
2. Copy the `eval/` directory and this file
3. Add `requirements.txt`: `gradio`, `gymnasium`, `numpy`
4. The app launches automatically

The adapter does NOT require `gradio` at import time. If gradio is
missing, the module imports successfully but `create_app()` raises
`ImportError` with installation instructions.

Or use programmatically:

```python
from sparse_parity.eval.adapters.huggingface import create_app

app = create_app()
app.launch(share=True)  # creates a public link
```

Key features:

- **Leaderboard tab**: shows baseline agents (Random, Greedy, Oracle) with mean reward, best method, best DMC, and discovery score
- **Try It tab**: enter a comma-separated method sequence, click Run, see experiment results and discovery grade
- **About tab**: explains the eval, grading rubric, and links to the repo
- **Answer Key tab**: shows the ground truth (rubric categories, method list, key facts)
- **Live updates**: your submission appears on the leaderboard table alongside baselines

## When to use which adapter

| Situation | Adapter |
|-----------|---------|
| Building a custom agent loop with the Anthropic API | `anthropic_tools.py` |
| Quick one-off eval of a Claude model | `run_anthropic_eval()` |
| Running evals through the Inspect framework | `inspect_task.py` |
| Publishing to PrimeIntellect Environment Hub | `primeintellect.py` |
| Interactive web leaderboard / HuggingFace Spaces | `huggingface.py` |
| Integrating with another platform | Use `anthropic_tools.py` as a template |

## Architecture

```
Platform (Anthropic API, Inspect, PrimeIntellect, HF Spaces, ...)
    |
    v
Adapter (anthropic_tools.py, inspect_task.py, primeintellect.py, huggingface.py)
    |
    v
SutroYaroEnv (env.py)  -- Gymnasium interface
    |
    v
Registry (registry.py)  -- method/challenge lookup
    |
    v
Backend (backends.py)   -- local, modal, or remote execution
    |
    v
Harness (harness.py)    -- actual experiment runner
```

## Links

- [Anthropic tool use docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Inspect framework docs](https://inspect.ai-safety-institute.org.uk/)
- [PrimeIntellect verifiers](https://github.com/PrimeIntellect-ai/verifiers)
- [PrimeIntellect community environments](https://github.com/PrimeIntellect-ai/community-environments)
- [HuggingFace Spaces](https://huggingface.co/spaces)
- [Gradio docs](https://www.gradio.app/docs)
- [SutroYaro eval README](../README.md)

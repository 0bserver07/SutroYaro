# Research as Navigation

## The thesis in one sentence

Research is primarily a navigation problem -- finding the right question and the right method -- and coding agents are the first tool that can do this autonomously because they read state, execute experiments, write results, and loop.

---

## ELI5: What does this mean?

Imagine you're lost in a forest and you need to find a specific tree. You have two options:

**Option A**: Walk in a straight line and hope you're going the right direction. (This is optimization -- pick a method, tune it, push harder.)

**Option B**: Climb the nearest hill, look around, pick a direction based on what you see, walk there, climb again, look again. (This is navigation -- survey the space, pick the most promising direction, check if it worked, repeat.)

Most AI research tools do Option A. They take one approach and optimize it. Our system does Option B. The agent reads what's been tried (climbs the hill), picks the next experiment (chooses a direction), runs it (walks there), and checks the result (looks around again).

The hard part is the hill-climbing, the navigation. Once you know what to try, running the experiment is easy. The 33 experiments in this project took 3 days of agent time. Figuring out *which* 33 to run, in *what order*, with *what comparisons* -- that's where all the value was.

---

## ELI a CS undergrad: The search space problem

Research on a well-defined problem (like sparse parity) is a search over a combinatorial space of:

- **Methods** (SGD, Fourier, GF(2), RL, genetic programming, ...)
- **Configurations** (n_bits, k_sparse, hidden, lr, batch_size, ...)
- **Comparisons** (what's the baseline? what metric? what's "better"?)

The space is too large to enumerate. C(16 methods, 1) * C(parameter combos) * C(baselines) gives thousands of possible experiments, most of which are uninformative.

Good research navigates this space efficiently:

1. **Prune**: If Hebbian learning fails, all local learning rules probably fail (they share the same limitation). Don't test 4 variants -- test 1, understand why it fails, skip the rest. Our agent tested all 4, but a smarter agent would have predicted the failure after the first one.

2. **Order**: Test GF(2) before optimizing SGD hyperparameters. If the problem is solvable in 500 microseconds algebraically, optimizing a 120ms neural network is wasted effort. The order of experiments matters more than the number.

3. **Compare**: "ARD improved 3.8%" is meaningless without knowing that W1 dominates 75% of float reads, capping any reordering gain at ~10%. The right comparison reveals the ceiling.

A coding agent can do all three because it can:

- Read previous results (navigate based on what's known)
- Read the search space definition (know what's available to try)
- Execute code (run the experiment and get real numbers)
- Write to files (record results for the next iteration)

A chatbot can suggest experiments. A notebook can run them. Only a coding agent can do both in a loop, accumulating knowledge across iterations.

---

## ELI a grad student: Why coding agents specifically?

There are many AI tools. Why are coding agents (Claude Code, Gemini CLI, Antigravity) better for research than chatbots, notebooks, or AutoML?

### What makes coding agents different

A coding agent has four capabilities that together make research navigation possible:

| Capability | What it enables | Chatbot? | Notebook? | AutoML? | Coding agent? |
|-----------|----------------|----------|-----------|---------|---------------|
| **Read files** | Survey what's known before acting | No | Manual | No | Yes |
| **Write files** | Record findings for future sessions | No | Manual | Limited | Yes |
| **Execute code** | Run experiments, get real numbers | No | Yes | Yes | Yes |
| **Loop autonomously** | Navigate the space without human input | No | No | Yes | Yes |

AutoML also loops, but it loops over a predefined search space with a fixed objective. It optimizes, it doesn't navigate. It can't decide "this whole approach is wrong, let me try something algebraic instead."

A coding agent can read DISCOVERIES.md, realize that all local learning rules failed because parity requires k-th order interactions, and decide to try GF(2) -- a completely different method family. That's navigation, not optimization.

### The customization argument

Coding agents are customizable through:

- **Protocol files** (AGENT.md, CLAUDE.md, GEMINI.md) -- tell the agent what to read, what loop to follow, what rules to obey
- **Skills and plugins** -- extend the agent with domain-specific capabilities
- **MCP servers** -- connect to external tools, databases, APIs
- **Search space definitions** -- bound what the agent can vary
- **Locked harnesses** -- prevent the agent from gaming metrics

This customization stack is what makes coding agents programmable research tools rather than chat assistants. You don't chat with the agent about what to try -- you program the navigation strategy in files, and the agent executes it.

The files are the program. The agent is the interpreter.

---

## ELI a PhD: Navigation vs optimization in experimental research

### The problem with optimization framing

Most automated research systems (NAS, AutoML, Bayesian optimization, evolutionary strategies) frame research as optimization: define a search space, define an objective, search. This works when the search space is known and the objective is fixed.

In real research, neither is true:

1. **The search space is not known in advance.** We started with SGD hyperparameter tuning. By experiment 17, we were doing Gaussian elimination over GF(2). No search space definition written before the project would have included "treat the problem as a linear system over the binary field."

2. **The objective shifts.** We started optimizing for accuracy, shifted to ARD (memory locality), then realized ARD doesn't model cache effects, built a cache simulator, and learned that total data movement matters more than average reuse distance. The metric evolved with understanding.

3. **The most valuable experiments are often negative.** Proving that all local learning rules fail on parity (because parity is invisible to methods limited to local statistics) is more valuable than finding a 3% ARD improvement. It eliminates an entire branch of the search tree.

### Navigation as the core research activity

Research navigation consists of:

**Survey**: What is known? What has been tried? What worked, what failed, and why?
This is DISCOVERIES.md, log.jsonl, and the experiment findings. The agent reads these before every cycle.

**Orient**: Given what's known, what's the most informative experiment to run next?
This is questions.yaml (dependency graph) and TODO.md (hypothesis queue). The agent picks the top unchecked item, but a better agent would reason about information gain.

**Test**: Run the experiment, get real numbers, compare against the right baseline.
This is the locked harness. The numbers must be trustworthy.

**Record**: Write down what happened -- not just the result, but the interpretation.
This is log.jsonl (machine-readable) and findings docs (human-readable). Failed experiments are findings too.

**Update**: Revise the map. What's now known? What questions are resolved? What new questions opened?
This is DISCOVERIES.md and questions.yaml being updated after each experiment.

This is Boyd's OODA loop (Observe, Orient, Decide, Act) applied to experimental science. The agent observes the current knowledge state, orients by reading the dependency graph, decides which experiment to run, acts by executing it, and the cycle repeats.

### Why coding agents fit this loop

The important property of coding agents is that they operate on the same substrate as the research artifacts. The agent reads Python files, writes Python files, runs Python experiments, and records results in files that it will read in the next iteration. There is no translation layer between the agent's "thinking" and the experimental work.

Compare this to:

- **A human researcher** navigates well but executes slowly. Reading 33 experiment reports, understanding the implications, designing the next experiment, writing the code, running it -- this takes days per iteration.

- **A chatbot** can reason about what to try next but cannot execute. It suggests experiments but someone else must run them. The feedback loop is broken by a human bottleneck.

- **AutoML** executes fast but navigates poorly. It explores a predefined space but cannot step outside it, cannot reason about why something failed, cannot decide that the entire framing is wrong.

- **A coding agent** navigates and executes in the same loop. It reads the knowledge base, designs an experiment, writes the code, runs it, interprets the results, updates the knowledge base, and picks the next experiment. The loop is tight and fully automated.

### The customization thesis

Coding agents fit research better than other automation tasks because their behavior is programmable through files:

**AGENT.md** is a program written in natural language that specifies the navigation strategy. It tells the agent what to read, what loop to follow, how to classify results, and when to stop. Different research groups can write different AGENT.md files for different navigation strategies, the same way different labs have different research methodologies.

**search_space.yaml** bounds the agent's exploration, preventing it from wandering into unproductive territory. But unlike AutoML's fixed search space, this file can be updated between cycles as understanding grows.

**DISCOVERIES.md** is the agent's memory. It persists across sessions, accumulates across researchers, and prevents the agent from repeating work. No other automated research system has a knowledge base that agents read before deciding what to do.

**Skills and plugins** extend the agent's capabilities. A skill that knows how to read arXiv papers, or query a citation database, or visualize training curves, makes the agent a better navigator. These are domain-specific navigation tools, not just general-purpose utilities.

The thesis is that coding agents are the first tool that can be *programmed to navigate research spaces* -- and that research, at its core, is navigation.

### What this predicts

If research is navigation and coding agents are the right tool, then:

1. **The quality of the navigation protocol matters more than the quality of any individual experiment.** A good AGENT.md that efficiently prunes the search space will outperform a bad one that exhaustively tries everything, even if individual experiments are identical.

2. **Multi-researcher systems outperform single-researcher systems** because they explore more of the space in parallel. This is why the peer research protocol exists -- not for throughput, but for coverage.

3. **Better orientation matters more than faster execution.** Teaching the agent to reason about which experiment is most informative (information gain, dependency resolution, search tree pruning) will matter more than making experiments run faster.

4. **The research artifacts (DISCOVERIES.md, log.jsonl, questions.yaml) are more valuable than the code.** The code runs one experiment; the artifacts guide all future experiments. This is why we lock the harness and invest in knowledge accumulation.

---

## Where we are now

The SutroYaro system implements this thesis at a basic level:

- The agent follows a fixed navigation strategy (top of TODO queue)
- The search space is bounded but static
- Knowledge accumulates in files
- Multiple researchers can contribute via PR

What's missing (and what would make the thesis stronger):

- **Adaptive navigation**: the agent should reason about which experiment is most informative, not just pick the next item in the queue
- **Cross-experiment reasoning**: "Hebbian failed because of X; therefore Predictive Coding will also fail because of X" -- the agent should prune without testing
- **Dynamic search space expansion**: when the agent hits a ceiling in one method family, it should propose new methods (like the jump from SGD to GF(2))
- **Information-theoretic experiment selection**: pick the experiment that maximizes expected information gain, not just the next one in line

These are hard problems. But the infrastructure to test them is in place.

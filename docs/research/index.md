# Research

Research notes and literature review for the Sutro Group.

## Research as Navigation

**[Research as Navigation](navigation-thesis.md)** is the thesis behind this project: research is primarily a navigation problem (finding the right question, method, comparison), and coding agents are the first tool that can navigate autonomously because they read state, execute experiments, write results, and loop. The page walks through this idea from ELI5 to PhD level, with examples from our 33 experiments.

## Autonomous Research Infrastructure

**[Peer Research Protocol](peer-research-protocol.md)** describes how SutroYaro runs autonomous, multi-researcher experiments. Multiple people use different AI tools (Claude Code, Gemini CLI, Codex CLI, OpenCode) on the same challenge. A locked evaluation harness ensures comparable results. A machine-readable experiment log accumulates findings across researchers.

Key infrastructure:

| Tool | What it does |
|------|-------------|
| `AGENT.md` | Machine-executable experiment loop for any AI agent |
| `src/harness.py` | Locked evaluation (5 methods, CLI). Agents cannot modify. |
| `bin/run-agent` | Tool-agnostic launcher with looped mode for overnight runs |
| `bin/analyze-log` | Progress report and chart from experiment log |
| `research/log.jsonl` | 33 experiments in machine-readable format |
| `research/search_space.yaml` | What the agent can vary, per challenge |

The protocol is challenge-agnostic: it works for sparse parity now and nanoGPT later. See the [full design doc](peer-research-protocol.md) for the nanoGPT migration proposal.

## Survey

**[Sparse Parity: A Practitioner's Field Guide](survey.md)** ranks all 33 experiments (16 Phase 1 + 17 Phase 2), provides a decision framework for picking methods, and documents the full AI research process including parallel agent dispatch.

## Topics

- [x] Sparse parity learning theory: [literature review](sparse-parity-literature.md)
- [x] Average Reuse Distance: theory, measurement, and [cache simulation](../findings/exp_cache_ard.md)
- [x] Forward-Forward algorithm: [tested, 25x worse ARD](../findings/exp_e_forward_forward.md)
- [x] Sign SGD: [solves k=5, 2x faster](../findings/exp_sign_sgd.md)
- [x] Per-layer forward-backward: [3.8% ARD improvement, converges identically](../findings/exp_c_perlayer_20bit.md)
- [x] Curriculum learning: [14.6x speedup on n=50](../findings/exp_curriculum.md)
- [x] Scaling frontier: [SGD breaks at n^k > 100K](../findings/exp_d_scaling.md)
- [x] Blank-slate approaches: [Fourier](../findings/exp_fourier.md), [evolutionary](../findings/exp_evolutionary.md), [feature selection](../findings/exp_feature_select.md)
- [x] [17 proposed approaches](proposed-approaches.md): all completed, results in [survey](survey.md)
- [ ] Deeper networks (5-10 layers) where FF's locality advantage may appear
- [ ] Hybrid approaches for k=8-9 (combinatorial search with pruning)

## Main Finding

!!! tip "For small k, sparse parity is a search problem, not a learning problem"
    Fourier/random search over C(n,k) subsets is 13-178x faster than SGD for k ≤ 7. Neural nets only become necessary when k ≥ 10 and C(n,k) explodes.

## Papers

| Paper | Year | Relevance | Link |
|-------|------|-----------|------|
| Hidden Progress in Deep Learning (Barak et al.) | 2022 | SGD learns sparse parity via hidden Fourier gap | [arxiv](https://arxiv.org/abs/2207.08799) |
| Matching SQ Lower Bound with Sign SGD (Kou et al.) | 2024 | Theoretically optimal sparse parity solver | [arxiv](https://arxiv.org/abs/2404.12376) |
| A Tale of Two Circuits (Merrill et al.) | 2023 | Grokking = sparse vs dense subnetwork competition | [arxiv](https://arxiv.org/abs/2303.11873) |
| GrokFast (Lee et al.) | 2024 | EMA gradient filter, counterproductive in our regime | [github](https://github.com/ironjr/grokfast) |
| Feature Learning Dynamics under Grokking | 2024 | NTK eigenfunctions align with secret indices | [openreview](https://openreview.net/forum?id=gciHssAM8A) |
| Bill Daly - Energy in GPUs | 2024 | Memory cost dominates energy | [YouTube](https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457) |
| DMC4ML (Ding et al.) | 2023 | Data Movement Complexity for ML | [arXiv](https://arxiv.org/abs/2312.14441) |
| Demmel - Communication-Avoiding Algorithms | 2013 | Lower bounds on data movement | [slides](https://simons.berkeley.edu/sites/default/files/docs/827/demmelslides.pdf) |

## Other Resources

| Resource | Type | Link |
|----------|------|------|
| Fitting Larger Networks into Memory | Article | [Medium](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9) |
| Sparse Parity background | Notebook | [NotebookLM](https://notebooklm.google.com/notebook/3eb7b53e-168f-409c-a679-2fb009119e2e) |
| Sparse Parity Optimization | Slides | [PDF](https://drive.google.com/file/d/1nC5KbckpwLjeGynuS4wPntBoClfDHyqY/view) |
| Hinton's Forward-Forward | Paper + Discussion | [Group notes](https://docs.google.com/document/d/1IdXRUhPRoWt8xLH1Y6iRWRx1g9-gbotFiiAnVixJYZY/edit?tab=t.0) |
| ARD Brainstorming | Gemini session | [Session](https://gemini.google.com/share/c99ec90874da) |
| parity-nn (minimal codebase) | GitHub | [Tsili42/parity-nn](https://github.com/Tsili42/parity-nn) |

## Concepts

### Average Reuse Distance (ARD)

Proxy metric for energy efficiency. Small ARD means data stays in fast cache. Large ARD means data must be fetched from external memory (HBM). Our CacheTracker extends this with LRU cache simulation for realistic estimates.

### Data Movement Complexity (DMC)

Added in v0.14.0 based on Yaroslav's [Knowledge Sprint #2](../google-docs/yaroslav-knowledge-sprint-2.md). DMC = sum of sqrt(stack_distance) for all float accesses (Ding et al., [arXiv:2312.14441](https://arxiv.org/abs/2312.14441)). Unlike ARD (which averages), DMC penalizes long-distance fetches sub-linearly through the square root, matching the physics of 2D chip layouts. The LRU cache lemma guarantees LRU is within 2x of optimal, so our LRU-based tracker gives realistic estimates.

Baseline (n=20/k=3): ARD 4,104 / DMC 300,298.

### The Roadmap

Yaroslav's [bigger picture](../google-docs/bigger-picture.md) defines three axes: **process** (agent harnesses), **metric** (ARD to DMC to GPU), **problem** (sparse parity to nanoGPT). Take small steps along one axis at a time. The final exam is energy-efficient nanoGPT training.

### The Giraffe Nerve Analogy

Backpropagation is like the recurrent laryngeal nerve in giraffes: it works but is inefficient because of the global memory access pattern. The brain uses ~20 Watts with local update rules. We want to find the AI equivalent.

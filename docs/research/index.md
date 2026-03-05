# Research

Research notes and literature review for the Sutro Group.

## Topics

- [x] Sparse parity learning theory: [literature review](sparse-parity-literature.md)
- [x] Average Reuse Distance: theory, measurement, and [cache simulation](../findings/exp_cache_ard.md)
- [x] Forward-Forward algorithm: [tested, 25x worse ARD](../findings/exp_e_forward_forward.md)
- [x] Sign SGD: [solves k=5, 2x faster](../findings/exp_sign_sgd.md)
- [x] Per-layer forward-backward: [3.8% ARD improvement, converges identically](../findings/exp_c_perlayer_20bit.md)
- [x] Curriculum learning: [14.6x speedup on n=50](../findings/exp_curriculum.md)
- [x] Scaling frontier: [SGD breaks at n^k > 100K](../findings/exp_d_scaling.md)
- [x] Blank-slate approaches: [Fourier](../findings/exp_fourier.md), [evolutionary](../findings/exp_evolutionary.md), [feature selection](../findings/exp_feature_select.md)
- [ ] [17 proposed approaches](proposed-approaches.md) — info-theoretic, algebraic, local learning, hardware-aware, alternative framings
- [ ] Deeper networks (5-10 layers) where FF's locality advantage may appear
- [ ] Predictive Coding on sparse parity
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

### The Giraffe Nerve Analogy

Backpropagation is like the recurrent laryngeal nerve in giraffes: it works but is inefficient because of the global memory access pattern. The brain uses ~20 Watts with local update rules. We want to find the AI equivalent.

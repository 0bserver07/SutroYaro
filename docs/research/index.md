# Research

Deep research notes and literature review for the Sutro Group.

## Topics to Investigate

- [ ] Average Reuse Distance -- theory and measurement
- [ ] Forward-Forward algorithm (Hinton 2022)
- [ ] Energy-efficient training methods survey
- [ ] Sparse parity learning theory
- [ ] Cache-aware neural network training
- [ ] Local learning rules vs backpropagation
- [ ] Predictive Coding vs Forward-Forward

## Key Papers / Resources

| Resource | Type | Link |
|----------|------|------|
| Bill Daly - Energy in GPUs | Talk | [YouTube](https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457) |
| Fitting Larger Networks into Memory | Article | [Medium](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9) |
| Sparse Parity background | Notebook | [NotebookLM](https://notebooklm.google.com/notebook/3eb7b53e-168f-409c-a679-2fb009119e2e) |
| Sparse Parity Optimization | Slides | [PDF](https://drive.google.com/file/d/1nC5KbckpwLjeGynuS4wPntBoClfDHyqY/view) |
| Hinton's Forward-Forward | Paper + Discussion | [Group notes](https://docs.google.com/document/d/1IdXRUhPRoWt8xLH1Y6iRWRx1g9-gbotFiiAnVixJYZY/edit?tab=t.0) |
| ARD Brainstorming | Gemini session | [Session](https://gemini.google.com/share/c99ec90874da) |

## Concepts

### Average Reuse Distance (ARD)

Proxy metric for energy efficiency. When ARD is small, data stays in fast, energy-efficient cache. When ARD is large, data must be fetched from expensive external memory (HBM).

### The Giraffe Nerve Analogy

Backpropagation is like the recurrent laryngeal nerve in giraffes -- it works but is wildly inefficient because of the global memory access pattern. The brain uses ~20 Watts with local update rules. We want to find the AI equivalent.

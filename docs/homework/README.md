# Homework

## Current: Meeting #8 (next)

**Drosophila of Learning** - Sparse Parity Challenge

Full spec: [challenge-1-sparse-parity.md](../google-docs/challenge-1-sparse-parity.md)

### Tasks
1. Generate training/testing datasets using random positive/negative numbers for XOR/parity
2. Build a neural net that solves the task (>90% accuracy)
3. Estimate energy via Average Reuse Distance (ARD)
4. Prompt AI to improve the algorithm's ARD
5. Scale to 3-bit parity with 17 noise "dirty" bits (20 total)

### Tips from the group
- Keep iteration time <1 second (training + eval)
- Change one thing at a time: correctness, then speed, then energy
- Priority order: correctness > wall-clock time > energy usage
- Checkpoint every correct + fast solution
- Use the [interactive ARD tutorial](https://ai.studio/apps/eca3f37a-175a-4713-bb17-622b24e17d3a)

### Reference implementation
- [Yaroslav Sprint 1 log](../google-docs/yaroslav-technical-sprint-1.md)
- [cybertronai/sutro repo](https://github.com/cybertronai/sutro) (sparse_parity_benchmark.py)
- [Colab notebook](https://colab.research.google.com/drive/1auWQjRgtrqyzef98wqq796sl927tCGMq)

---

## Past Homework Archive

### Meeting #5 (16 Feb 26) - Karpathy Names Task

**Goal**: Optimize a character-level model for energy efficiency.

**Setup**:
- Take 1000 random names from Karpathy's [makemore/names.txt](https://github.com/karpathy/makemore/blob/master/names.txt) for training
- Take another 1000 random names and predict last 3 characters of each

**Tasks**:
1. Create baseline: obtain baseline accuracy and total operation count
2. Improve baseline: reduce total operations without reducing accuracy
3. Share tips and lessons

**Reference**: [Intelligence_Per_Joule.pdf](https://drive.google.com/open?id=1vyvElj7aTFZYwNpA1mzHAaLSJegmAson&usp=drive_fs)

**Emmett's approach**: Pure-Python GPT (no dependencies), single transformer layer, 16-dim embeddings, 4 attention heads. Reduced memory from 80MB to 35MB using Aster agentic loop (8 iterations). [Full implementation](https://docs.google.com/document/d/1DAwx_gohi6tomMPkb_fETAIuxIyHgLtC5OPD_qpGpqg/edit?tab=t.0)

---

### Meeting #2 (26 Jan 26) - Forward-Forward Algorithm

**Goal**: Understand Hinton's Forward-Forward as an alternative to backprop.

**Reading**: [Hinton's Forward-Forward Paper Discussion](https://docs.google.com/document/d/1IdXRUhPRoWt8xLH1Y6iRWRx1g9-gbotFiiAnVixJYZY/edit?tab=t.0)

**Key concepts to study**:
- Two forward passes (positive/negative) instead of forward+backward
- Greedy layer-wise learning (each layer has its own objective)
- Goodness function = sum of squared ReLU activations
- The "negative data" problem for complex domains

**Exercises**:
1. Reproduce MNIST classification with Forward-Forward in PyTorch
2. Visualize negative data -- try random noise vs permuted pixels, observe accuracy impact
3. Try the "Sandwich" method: FF for first 3 layers (feature extractor), then standard softmax on top
4. Run permutation invariance test (shuffle MNIST pixels, compare FF vs ConvNet)

**Follow-up reading**:
- Hinton's "Mortal Computation" (philosophy behind FF)
- Compare with Predictive Coding (minimizes prediction error vs FF's goodness)
- Recurrent Forward-Forward extension (addresses greedy limitation)

**Jamie Simon's results**: [implementation doc](https://docs.google.com/document/d/1u8pIWg2iWc9R-dQgXz2Yt6xgF2fWP0GbDipbIxF1v1Y/edit?tab=t.0)

---

### Meeting #3 (02 Feb 26) - Joules Measuring

**Goal**: Set up tooling to measure energy consumption of training.

**Tasks**:
1. Get the Joules-measuring [Colab notebook](https://colab.research.google.com/drive/1ctren0aejK4KI9AYclUrGbDYZrDqSiJS#scrollTo=kdhH19I9XvDZ) working
2. Try Modal workflow (Barak's approach) or Colab workflow (Yaroslav's approach)
3. Measure joules for a simple training loop

---

### Meeting #1 (19 Jan 26) - Energy-Efficient Training Intro

**Goal**: Orientation. Understand why energy efficiency matters for AI training.

**Key ideas**:
- Memory cost is the biggest energy contributor (Bill Daly [talk](https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457))
- Local registers ~5pJ vs HBM ~640pJ (128x difference)
- Backprop is like the giraffe's recurrent laryngeal nerve -- works but inefficient
- Brain runs at ~20 Watts, uses local update rules
- "Nerd snipe" proposal: train a model on smartphone via WebGPU using minimum joules

**Notes**: [sutro meeting #1](https://docs.google.com/document/d/1ZsH26hVvbZBOshwA1KgdX5AK5zw9W0CzqZuXLa5fIlo/edit?tab=t.0)

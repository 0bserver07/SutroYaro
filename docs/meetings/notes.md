# Detailed Meeting Notes

## Meeting #1, 19 Jan 26 - Energy-Efficient Training

**Location**: SPC main floor · [Full notes](../google-docs/meeting-1-energy-intro.md) · [Google Doc](https://docs.google.com/document/d/1ZsH26hVvbZBOshwA1KgdX5AK5zw9W0CzqZuXLa5fIlo/edit?tab=t.0)

Orientation meeting. Introductions and backgrounds. Concepts introduced:

- Memory cost is the largest energy contributor (Bill Daly [talk](https://youtu.be/rsxCZAE8QNA?si=8-kIJ1MuhxChRLgW&t=2457))
- Local registers ~5pJ vs HBM ~640pJ
- Backprop is like the giraffe's recurrent laryngeal nerve -- works but inefficient
- "Nerd snipe" proposal: train a model on smartphone via WebGPU using minimum joules
- WebGPU exposes memory hierarchy (Registers -> Shared -> Global)

!!! quote "Takeaway"
    Yaroslav beat Google in 2018 DawnBench (fastest ImageNet training) not through superior intelligence but 3 months optimizing AWS infrastructure for 10-second restart cycles versus Google's 10+ minutes.

---

## Meeting #2, 26 Jan 26 - Forward-Forward Algorithm

**Location**: Accel board room · [Full notes](../google-docs/meeting-2-forward-forward.md) · [Google Doc](https://docs.google.com/document/d/1IdXRUhPRoWt8xLH1Y6iRWRx1g9-gbotFiiAnVixJYZY/edit?tab=t.0)

Discussion of Hinton's Forward-Forward paper. See also: [Exp E - Forward-Forward findings](../findings/exp_e_forward_forward.md).

- Two forward passes (positive/negative) replace forward+backward
- Greedy layer-wise learning: each layer has its own objective
- Goodness = sum of squared ReLU activations
- Negative data generation is the hard problem for complex domains
- Jamie Simon shared implementation results

---

## Meeting #3, 02 Feb 26 - Joules Measuring

**Location**: SPC

Tooling session.

- Barak demonstrated Modal workflow
- Yaroslav demonstrated Colab workflow
- [Joules-measuring notebook](https://colab.research.google.com/drive/1ctren0aejK4KI9AYclUrGbDYZrDqSiJS#scrollTo=kdhH19I9XvDZ)

---

## Meeting #4, 09 Feb 26 - From Beauty to Joules

**Location**: Palmer Square

Presentation: [From_Beauty_to_Joules.pdf](https://drive.google.com/open?id=1T2jTdXnS5L7JGri5clrCz4hRslxIr6Vg&usp=drive_fs)

---

## Meeting #5, 16 Feb 26 - Intelligence Per Joule

Presentation: [Intelligence_Per_Joule.pdf](https://drive.google.com/open?id=1vyvElj7aTFZYwNpA1mzHAaLSJegmAson&usp=drive_fs)

**Karpathy Names Task** introduced:

- Take 1000 random names from [makemore/names.txt](https://github.com/karpathy/makemore/blob/master/names.txt)
- Predict last 3 characters of 1000 test names
- Baseline accuracy + total operations -> optimize

---

## Meeting #6, 23 Feb 26 - Presentations

[Full notes](../google-docs/meeting-6-notes.md) · [Google Doc](https://docs.google.com/document/d/1OXd_-RweVbHzjqTA2UF05mEs8iJZQiqKeU3QHU_DMdc/edit?tab=t.0)

- **Germaine**: [presentation video](https://drive.google.com/file/d/1BzZLWCveiXRAXj2nAsYYDoPnPe8lDDki/view?usp=sharing) — truncated backprop, 19% energy reduction, 27% intelligence-per-joule improvement
- **Emmett**: Pure-Python GPT, reduced memory 80MB -> 35MB with Aster ([local](../google-docs/meeting-6-emmett-results.md) · [Google Doc](https://docs.google.com/document/d/1DAwx_gohi6tomMPkb_fETAIuxIyHgLtC5OPD_qpGpqg/edit?tab=t.0))
- Yaroslav presented pebbling games, energy hierarchy, "drosophila of learning" concept
- Key outcome: 3-minute MicroGPT iteration too slow — need sub-1-second task

---

## Meeting #7, 02 Mar 26 - Sparse Parity

[Full notes](../google-docs/meeting-7-notes.md) · [Google Doc](https://docs.google.com/document/d/1M5cJVDTb3AbCz1w19O333FuQrSIoQHW8y6pJK4KwEO4/edit?tab=t.0)

- Yaroslav presented [Technical Sprint 1](../google-docs/yaroslav-technical-sprint-1.md) results — 2.5hr sprint, ARD metric, gradient fusion (16% cache reuse improvement)
- Andy attempted better chat tooling ([codeberg](https://codeberg.org/zh4ng/chat))
- Michael showed [Pebbling Game implementation](https://claude.ai/public/artifacts/647d9d58-211c-4f32-83e6-500353f47d86)
- Homework assigned: [Challenge #1: Sparse Parity](../google-docs/challenge-1-sparse-parity.md) ("Drosophila of Learning")

See also: [Research overview](../research/index.md) for all experiment results building on this challenge.

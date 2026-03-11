# Experiment: {Title}

**Date**: YYYY-MM-DD
**Author**: {your name}
**Status**: SUCCESS | PARTIAL | FAILED
**Answers**: {Which open question from DISCOVERIES.md, if any}

## Hypothesis

{One sentence: "If we do X, then Y will happen because Z."}

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20 |
| k_sparse | 3 |
| hidden | 200 |
| lr | 0.1 |
| wd | 0.01 |
| batch_size | 32 |
| n_train | 1000 |
| max_epochs | 200 |
| seed | 42 |
| method | {what you changed} |

## Results

| Metric | Baseline | Experiment |
|--------|----------|------------|
| Test accuracy | | |
| Wall time | | |
| ARD | | |
| DMC | | |

## Analysis

### What worked

- {bullet points}

### What didn't work

- {bullet points}

### Surprise

{The one thing you didn't expect}

## Open Questions

- {Question that came up during this experiment}
- {Specific enough to be tested next}

## Files

- Experiment code: `src/sparse_parity/experiments/{name}.py` (if applicable)
- Results: `results/{name}/results.json` (if applicable)
- This document: `findings/{name}.md`

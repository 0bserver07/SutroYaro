# Task 1: Switch to Data Movement Complexity (DMC)

**Priority**: HIGH
**Status**: DONE
**Source**: Yaroslav Knowledge Sprint #2, Meeting #8

## Context

Yaroslav's Knowledge Sprint #2 concluded:

- ARD is "likely sufficient" but DMC is "likely a better metric"
- DMC corresponds directly to energy cost for a specific 2D memory layout
- Connects to existing literature: Ding's DMC4ML paper (arXiv:2312.14441)
- LRU cache lemma: LRU is within 2x of optimal, so just assume LRU

Our `CacheTracker` already simulates LRU cache. Extend it to compute DMC alongside ARD.

## Tasks

- [ ] Read Ding's DMC4ML paper and extract the formula
- [ ] Extend `cache_tracker.py` to compute DMC alongside ARD
- [ ] Update experiment template to report both metrics
- [ ] Re-run baseline (fast.py) with DMC and compare against ARD
- [ ] Update DISCOVERIES.md with DMC baseline numbers

## References

- Yaroslav Knowledge Sprint #2: docs/google-docs/yaroslav-knowledge-sprint-2.md
- Ding paper: https://arxiv.org/abs/2312.14441
- Current CacheTracker: src/sparse_parity/cache_tracker.py
- The Bigger Picture: axis 2 (metric) is where this fits

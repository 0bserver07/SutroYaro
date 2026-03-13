# Task 2: Use Stack Distance (bytes) instead of Reuse Distance (instructions)

**Priority**: HIGH
**Status**: DONE (already implemented)
**Source**: Yaroslav Knowledge Sprint #2

## Context

From Yaroslav's notes: "must rely on bytes touched (Stack Distance) rather than instructions elapsed (Reuse Distance)."

Our MemTracker counts instruction-level reuse distance (how many other accesses happen between two accesses to the same tensor). Stack distance counts unique bytes touched between reuses. This is more physically meaningful because:

- A 200x20 W1 matrix costs more to evict than a 200x1 bias vector
- Instruction count treats them the same
- Bytes touched reflects actual cache pressure

## Resolution

Already implemented. The MemTracker clock advances by buffer SIZE (floats), not operation count (see tracker.py line 8). So `distance = self.clock - self._write_time[name]` already measures the number of floats accessed between reuses, which is stack distance in float units. Multiply by 4 for bytes.

This was a deliberate design choice from the beginning. No code change needed.

## References

- Yaroslav Knowledge Sprint #2: section "Ranking heuristics for LRU cache reuse"
- Current MemTracker: src/sparse_parity/tracker.py
- Herb Sutter talk on memory wall: https://nwcpp.org/talks/2007/Machine_Architecture_-_NWCPP.pdf

# TrackedArray: Auto-instrumented DMD Tracking

## Problem

The existing MemTracker requires manual `tracker.read()` / `tracker.write()` calls placed throughout experiment code. This is error-prone: the GF(2) experiment reported DMC of 8,607 but the actual Gaussian elimination (row swaps, XOR operations) had zero tracking calls.

Manual instrumentation also creates a barrier for competition submissions. If someone submits a new algorithm, we can't trust them to instrument correctly, and we shouldn't require them to.

## The metric: Granular DMD (Ding et al., arXiv:2312.14441)

Data Movement Distance (DMD) measures the cost of each memory access using an LRU stack model (Definition 2.1 in the paper).

Every float lives in an LRU stack ordered by recency of writes. When a float is read, its **stack distance** is its 1-indexed position in the stack. Its DMD is `sqrt(stack_distance)`. The total DMC of an algorithm is the sum of all DMDs.

Key rules:
- **Writes** move the element to the top of the stack (position 1).
- **Reads** observe the element's position but do not move it. Reading data does not change where it lives in the memory hierarchy.
- **Cold misses** (first access to an element not yet in the stack) have distance = `len(stack) + 1`.

From the paper: "in `abbbca`, the reuse distance of the second `a` is 3. Its DMD is sqrt(3)."

### Worked example: `(a+b)+a`

Arrays of size 1: `a=[1.0]`, `b=[5.0]`. Each array is a single float.

```
Step 1: write a       stack = [a]           a is new, cold miss
Step 2: write b       stack = [b, a]        b is new, cold miss

Step 3: compute a + b
  read a:  a is at position 2    dist = 2    stack unchanged [b, a]
  read b:  b is at position 1    dist = 1    stack unchanged [b, a]
  write c: c goes to top         stack = [c, b, a]

Step 4: compute c + a
  read c:  c is at position 1    dist = 1    stack unchanged [c, b, a]
  read a:  a is at position 3    dist = 3    stack unchanged [c, b, a]
  write d: d goes to top         stack = [d, c, b, a]
```

Read DMD calculation:

| Read | Stack distance | DMD contribution |
|------|---------------|-----------------|
| a (in a+b) | 2 | sqrt(2) = 1.414 |
| b (in a+b) | 1 | sqrt(1) = 1.000 |
| c (in c+a) | 1 | sqrt(1) = 1.000 |
| a (in c+a) | 3 | sqrt(3) = 1.732 |

**Read DMD = 1.414 + 1.000 + 1.000 + 1.732 = 5.146**

Note: the second read of `a` has distance 3 (not 6). There are only 3 elements in the stack, so `a` cannot be deeper than position 3. The old clock-based model inflated this to 6.

### Previous metric (MemTracker, deprecated for this purpose)

The old MemTracker used `distance = clock - write_time[name]` where the clock advanced by buffer size on every read and write. This is not the paper's definition:
- It measured from last **write** only, ignoring reads
- It used a monotonic clock instead of LRU stack positions
- It treated entire buffers as single units instead of tracking per-element

## Solution

Two components:

### LRUStackTracker

`src/sparse_parity/lru_tracker.py` -- true per-element LRU stack tracker. Each float is identified by `(buffer_name, index)`. Maintains an LRU stack of all elements. On write, elements are pushed to top. On read, stack positions are observed without modification.

### TrackedArray (ndarray subclass)

`src/sparse_parity/tracked_numpy.py` -- wraps `np.ndarray` to automatically call `tracker.write()` and `tracker.read()` on every operation. Works with both the old MemTracker and the new LRUStackTracker (same API).

Operations are intercepted at three levels:

1. **`__array_ufunc__`** -- catches all ufuncs: `+`, `-`, `*`, `^`, `==`, `<`, etc. For each TrackedArray input, records a `tracker.read()`. For the output, creates a new TrackedArray and records a `tracker.write()`. In-place ops (with `out=`) write back to the existing buffer name.

2. **`__array_function__`** -- catches numpy functions: `np.where`, `np.prod`, `np.sum`, `np.all`, `np.sort`, `np.concatenate`, etc. A default handler covers any unregistered function.

3. **`__getitem__` / `__setitem__`** -- catches indexing, slicing, and fancy indexing. `__getitem__` records a read sized to the actual slice (not the whole array). `__setitem__` records a read of the source and a write to the target slice.

### tracking_context (context manager)

Solves a bootstrapping problem: `np.zeros(shape)` has no array arguments, so `__array_function__` never fires. Inside a `tracking_context`, constructors (`np.zeros`, `np.ones`, `np.empty`) are monkey-patched to return TrackedArrays. Patches revert on exit.

### Propagation

Any array derived from a TrackedArray is itself a TrackedArray on the same tracker. Wrapping initial inputs is sufficient.

## Usage

```python
from sparse_parity.tracked_numpy import TrackedArray, tracking_context
from sparse_parity.lru_tracker import LRUStackTracker

tracker = LRUStackTracker()
with tracking_context(tracker):
    A = TrackedArray(A_raw, "A", tracker)
    b = TrackedArray(b_raw, "b", tracker)
    solution, rank = gf2_gauss_elim(A.copy(), b.copy())

s = tracker.summary()
print(s["read_dmd"])      # DMD from reads only
print(s["granular_dmd"])  # DMD from all accesses (reads + writes)
```

Zero changes inside `gf2_gauss_elim`.

## GF(2) results

| Method | Read DMD | Notes |
|--------|----------|-------|
| Manual harness (I/O only) | 8,607 | Only tracks data conversion, misses elimination |
| Yad's honest estimate | 189,056 | Manual count of row operations |
| LRUStackTracker auto | 203,444 | Per-element LRU stack, all ops tracked |

## Limitations

- **Overhead**: LRUStackTracker is O(n) per element access where n = stack size. GF(2) with n=20 takes ~12s. Not for production, only measurement.
- **Pure-python loops**: Scalar access like `aug[row, col] == 1` generates per-element tracking events. Correct but verbose.
- **Non-numpy code**: Values extracted as plain python (`int(arr[0])`) exit the tracking world.
- **Constructor coverage**: Only `np.zeros`, `np.ones`, `np.empty` are patched inside `tracking_context`.

## Files

- `src/sparse_parity/lru_tracker.py` -- LRUStackTracker (true per-element LRU stack)
- `src/sparse_parity/tracked_numpy.py` -- TrackedArray + tracking_context
- `src/sparse_parity/tracker.py` -- MemTracker (old clock-based, kept for backward compat)
- `tests/test_tracked_numpy.py` -- 29 tests

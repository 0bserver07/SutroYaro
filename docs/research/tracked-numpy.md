# Auto-instrumented DMD Tracking

TrackedArray wraps `np.ndarray` so that every numpy operation -- arithmetic, indexing, slicing, function calls -- automatically records memory reads and writes on an LRU stack tracker. You wrap your inputs, run unmodified numpy code, and get per-element Data Movement Distance (DMD) metrics out the other end.

This replaces manual `tracker.read()` / `tracker.write()` calls, which were error-prone and required instrumenting every algorithm by hand.

## Quick start

```python
import numpy as np
from sparse_parity.tracked_numpy import TrackedArray, tracking_context
from sparse_parity.lru_tracker import LRUStackTracker

tracker = LRUStackTracker()

# Raw numpy arrays (your algorithm's inputs)
A_raw = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)
b_raw = np.array([1, 0], dtype=np.int8)

with tracking_context(tracker):
    A = TrackedArray(A_raw, "A", tracker)
    b = TrackedArray(b_raw, "b", tracker)
    # Run any numpy code -- all ops auto-tracked
    result = A @ b

s = tracker.summary()
print(f"Read DMD: {s['read_dmd']:.1f}")
print(f"Total DMD: {s['granular_dmd']:.1f}")
```

The `tracking_context` is needed so that constructors like `np.zeros` and `np.ones` inside your algorithm return TrackedArrays instead of plain arrays. Any array derived from a TrackedArray (through ufuncs, indexing, or numpy functions) is itself a TrackedArray on the same tracker.

## How it works

Three interception layers capture all numpy operations:

1. **`__array_ufunc__`** catches all ufuncs: `+`, `-`, `*`, `^`, `==`, `<`, etc. Each TrackedArray input triggers a `tracker.read()`. The output gets a `tracker.write()`. In-place ops (`out=`) write back to the existing buffer name.

2. **`__array_function__`** catches numpy functions: `np.where`, `np.prod`, `np.sum`, `np.all`, `np.sort`, `np.concatenate`, etc. A default handler covers any unregistered function.

3. **`__getitem__` / `__setitem__`** catches indexing, slicing, and fancy indexing. `__getitem__` records a read sized to the actual slice (not the whole array). `__setitem__` records a read of the source and a write to the target slice.

The `tracking_context` context manager solves a bootstrapping problem: `np.zeros(shape)` has no array arguments, so `__array_function__` never fires. Inside the context, constructors (`np.zeros`, `np.ones`, `np.empty`) are monkey-patched to return TrackedArrays. Patches revert on exit.

## The metric: Data Movement Distance

Data Movement Distance (DMD) is defined in Ding et al. (arXiv:2312.14441, Definition 2.1). It measures the cost of each memory access using an LRU stack model.

Every float lives in an LRU stack ordered by recency of writes. When a float is accessed:

- **Writes** move the element to the top of the stack (position 1).
- **Reads** observe the element's position but do not move it.
- **Cold misses** (first access to an element not yet in the stack) have distance = `len(stack) + 1`.

The DMD of a single access is `sqrt(stack_distance)`. The total Data Movement Complexity (DMC) of an algorithm is the sum of all DMDs.

From the paper: "in `abbbca`, the reuse distance of the second `a` is 3. Its DMD is sqrt(3)."

### Worked example: `(a+b)+a`

Arrays of size 1: `a=[1.0]`, `b=[5.0]`. Each array is a single float.

```
Step 1: write a       stack = [a]             free
Step 2: write b       stack = [a, b]          free

Step 3: compute a + b
  read a:  a is at position 2    dist = 2    stack unchanged [a, b]
  read b:  b is at position 1    dist = 1    stack unchanged [a, b]
  write c: c goes to top         stack = [a, b, c]

Step 4: compute c + a
  read c:  c is at position 1    dist = 1    stack unchanged [a, b, c]
  read a:  a is at position 3    dist = 3    stack unchanged [a, b, c]
  write d: d goes to top         stack = [a, b, c, d]
```

Read DMD calculation:

| Read | Stack distance | DMD contribution |
|------|---------------|-----------------|
| a (in a+b) | 2 | sqrt(2) = 1.414 |
| b (in a+b) | 1 | sqrt(1) = 1.000 |
| c (in c+a) | 1 | sqrt(1) = 1.000 |
| a (in c+a) | 3 | sqrt(3) = 1.732 |

**Read DMD = 1.414 + 1.000 + 1.000 + 1.732 = 5.146**

Note: the second read of `a` has distance 3 (not 6). There are only 3 elements in the stack, so `a` cannot be deeper than position 3.

## API reference

### TrackedArray

```python
TrackedArray(data, name, tracker)
```

- `data` -- a numpy array (or anything `np.asarray` accepts)
- `name` -- string identifier for this buffer (used in per-buffer reports)
- `tracker` -- an `LRUStackTracker` instance

The constructor calls `tracker.write(name, size)` to register the initial data.

### tracking_context

```python
with tracking_context(tracker):
    # np.zeros, np.ones, np.empty return TrackedArrays here
    ...
```

Monkey-patches numpy constructors for the duration of the block. Thread-safe (uses `threading.local`).

### LRUStackTracker

```python
tracker = LRUStackTracker()
```

Methods:

- `tracker.write(name, size)` -- write `size` floats, pushing each to the top of the LRU stack.
- `tracker.read(name, size)` -- read `size` floats, observing stack positions without moving them. Returns list of per-element stack distances.
- `tracker.summary()` -- returns a dict with all metrics.
- `tracker.report()` -- prints a formatted report to stdout.

### summary() fields

| Field | Type | Meaning |
|-------|------|---------|
| `dmd` | float | Total DMD (reads only -- writes are free) |
| `reads` | int | Number of read operations (not elements) |
| `writes` | int | Number of write operations (not elements) |
| `stack_size` | int | Current number of elements in the LRU stack |
| `per_buffer` | dict | Per-buffer breakdown with `avg_dist`, `min_dist`, `max_dist`, `read_count`, `dmd` |

## Limitations

- **Performance overhead.** LRUStackTracker is O(n) per element access where n is the stack size. GF(2) Gaussian elimination with n=20 takes about 12 seconds under tracking. This is a measurement tool, not a production runtime.
- **Pure-python loops.** Scalar access like `aug[row, col] == 1` generates per-element tracking events. Correct but slow and verbose in event logs.
- **Non-numpy code.** Values extracted as plain Python scalars (`int(arr[0])`) leave the tracking world. Subsequent operations on those scalars are not tracked.
- **Constructor coverage.** Only `np.zeros`, `np.ones`, and `np.empty` are patched inside `tracking_context`. Other constructors (`np.arange`, `np.linspace`, etc.) return plain arrays.

## Files

| File | What it is |
|------|-----------|
| `src/sparse_parity/lru_tracker.py` | LRUStackTracker (per-element LRU stack) |
| `src/sparse_parity/tracked_numpy.py` | TrackedArray + tracking_context |
| `src/sparse_parity/tracker.py` | MemTracker (old clock-based, kept for backward compat) |
| `tests/test_tracked_numpy.py` | 29 tests covering wrapper mechanics, indexing, numpy functions, LRU metrics, and GF(2) integration |

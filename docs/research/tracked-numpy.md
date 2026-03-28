# Auto-instrumented DMD Tracking

TrackedArray wraps `np.ndarray` so that every numpy operation automatically records memory reads and writes on an LRU stack tracker. Wrap your inputs, run unmodified numpy code, and get Data Movement Distance (DMD) out the other end. No manual instrumentation needed.

## Quick start

```python
import numpy as np
from sparse_parity.tracked_numpy import TrackedArray, tracking_context
from sparse_parity.lru_tracker import LRUStackTracker

tracker = LRUStackTracker()
with tracking_context(tracker):
    A = TrackedArray(A_raw, "A", tracker)
    b = TrackedArray(b_raw, "b", tracker)
    result = my_algorithm(A, b)  # unmodified code

print(tracker.summary()["dmd"])
```

## The metric: DMD (Ding et al., arXiv:2312.14441)

Every float lives in an LRU stack ordered by recency of writes:

- **Writes** move the element to the top of the stack. Writes are free (no DMD cost). Inputs arrive pre-loaded on the stack.
- **Reads** observe the element's position but do not move it. DMD = `sqrt(stack_distance)`.
- **Total DMD** of an algorithm = sum of all read DMDs.

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

| Read | Stack distance | DMD contribution |
|------|---------------|-----------------|
| a (in a+b) | 2 | sqrt(2) = 1.414 |
| b (in a+b) | 1 | sqrt(1) = 1.000 |
| c (in c+a) | 1 | sqrt(1) = 1.000 |
| a (in c+a) | 3 | sqrt(3) = 1.732 |

**DMD = 1.414 + 1.000 + 1.000 + 1.732 = 5.146**

## Tracked operations

Every numpy operation on a TrackedArray is decomposed into reads (inputs) and writes (outputs). The table below shows how each operation class maps to tracker calls. The DMD cost comes entirely from the reads.

### Elementwise operations (ufuncs)

`c = a + b`, `c = a * b`, `c = a ^ b`, `a == 1`, etc.

| Step | Tracker call | Size |
|------|-------------|------|
| Read each input | `read(a, len(a))`, `read(b, len(b))` | Full array |
| Write result | `write(c, len(c))` | Full array |

DMD = sum of sqrt(stack_distance) for each element of `a` and `b`. The write of `c` is free.

For in-place operations (`np.add(a, b, out=c)`), the write goes to the existing buffer `c` instead of creating a new one.

### Matrix multiply (`@`, `np.dot`, `np.matmul`)

`C = A @ B` where A is (m, n) and B is (n, l).

| Step | Tracker call | Size |
|------|-------------|------|
| Read A | `read("A", m*n)` | All elements of A |
| Read B | `read("B", n*l)` | All elements of B |
| Write C | `write("C", m*l)` | Result matrix |

DMD = sum of sqrt(stack_distance) for each element of A and B. The cost depends on where A and B sit in the stack relative to other data.

**Theoretical DMD for naive n x n matmul** (Smith, Goldfarb, Ding 2022, arXiv:2203.02536):

```
DMD_naive = ~(n^4)
```

The exact formula is `(n^3 * sqrt(2n)) + (n^3 - 2n^2 + n) * sqrt(n^2 + 2n)`. The n^4 scaling comes from n^3 multiply-add operations, each paying ~sqrt(n) for accessing elements that are ~n positions deep in the stack. Better algorithms reduce this:

| Algorithm | Time | DMD | Source |
|-----------|------|-----|--------|
| Naive (triple-loop) | O(n^3) | ~n^4 | Smith et al. 2022, Sec 5.1 |
| Tiled (D x D tiles) | O(n^3) | ~n^4/D + n^3*sqrt(D) | Theorem 3 |
| Recursive | O(n^3) | ~13.5 * n^3.5 | Theorem 4 |
| Recursive + temp reuse | O(n^3) | ~11.9 * n^3.33 | Theorem 5 |
| Strassen | O(n^2.8) | ~6.5 * n^3.4 | Theorem 6 |
| Strassen + temp reuse | O(n^2.8) | ~15.4 * n^3.23 | Theorem 7 |

Key insight: all six algorithms have the same or similar time complexity, but DMD reveals large differences in data movement cost. Recursive MM with temp reuse achieves n^3.33 vs naive's n^4.

**What TrackedArray measures**: numpy implements matmul as a single call, so TrackedArray records one bulk read of A and one of B. This corresponds to the naive access pattern. It does not model tiled or recursive inner-loop reuse. For small matrices (as in our experiments), this is accurate. For large matrices, the true DMD may be lower if numpy's BLAS backend uses tiled or recursive algorithms internally.

### Indexing and slicing

`row = A[i]`, `block = A[2:5, :]`, `val = A[i, j]`

| Step | Tracker call | Size |
|------|-------------|------|
| Read source | `read("A", slice_size)` | Size of the actual slice, not the whole array |
| Write result | `write("slice", slice_size)` | The extracted slice |

Scalar access (`A[i, j]`) records a read of size 1. Row access (`A[i]`) records a read of size n. This matters for GF(2) Gaussian elimination where the pivot search does many scalar reads.

### Assignment (`__setitem__`)

`A[i] = row`, `A[0:3] = B[0:3]`

| Step | Tracker call | Size |
|------|-------------|------|
| Read source value | `read("row", len(row))` | If source is TrackedArray |
| Write target slice | `write("A", slice_size)` | Size of the written region |

### Reductions (`np.sum`, `np.prod`, `np.mean`, `np.all`)

`s = np.sum(A)`, `p = np.prod(A, axis=1)`

| Step | Tracker call | Size |
|------|-------------|------|
| Read input | `read("A", len(A))` | Full array |
| Write result | `write("result", result_size)` | Scalar or reduced array |

### Constructors (`np.zeros`, `np.ones`, `np.empty`)

Inside a `tracking_context`, these return TrackedArrays:

| Step | Tracker call | Size |
|------|-------------|------|
| Write new array | `write("zeros", size)` | Full array |

No read. The data is freshly created and placed on the stack.

### Zero-cost operations

These do **not** record any reads or writes:

- **Transpose** (`.T`): Returns a view sharing the same buffer name. No data movement.
- **`np.zeros_like(a)`**: Creates a new buffer but does not read `a` (only inspects shape/dtype).

### Copy and type conversion

`b = a.copy()`, `b = a.astype(np.uint8)`

| Step | Tracker call | Size |
|------|-------------|------|
| Read source | `read("a", len(a))` | Full array |
| Write copy | `write("copy", len(a))` | Full array |

## How it works

Three interception layers capture all numpy operations:

1. **`__array_ufunc__`** catches ufuncs (`+`, `-`, `*`, `^`, `==`, `<`, `@`, etc.). Each TrackedArray input triggers a `tracker.read()`. The output gets a `tracker.write()`.

2. **`__array_function__`** catches numpy functions (`np.where`, `np.prod`, `np.sum`, `np.sort`, `np.concatenate`, etc.). A default handler records reads for all TrackedArray inputs and wraps the output.

3. **`__getitem__` / `__setitem__`** catches indexing and slicing. Reads are sized to the actual slice accessed.

The `tracking_context` context manager monkey-patches `np.zeros`, `np.ones`, and `np.empty` to return TrackedArrays. Patches revert on exit. Any array derived from a TrackedArray inherits tracking automatically.

## GF(2) results

| Method | DMD | Notes |
|--------|-----|-------|
| Manual harness (I/O only) | 8,607 | Missed elimination loop |
| Yad's honest estimate | 189,056 | Manual count of row operations |
| TrackedArray auto | ~203,000 | All ops tracked |

## API reference

### TrackedArray

```python
TrackedArray(data, name, tracker)
```

- `data` -- a numpy array
- `name` -- string identifier for this buffer
- `tracker` -- an `LRUStackTracker` instance

The constructor calls `tracker.write(name, size)` to place the data on the stack.

### tracking_context

```python
with tracking_context(tracker):
    ...  # np.zeros, np.ones, np.empty return TrackedArrays
```

### LRUStackTracker

- `tracker.write(name, size)` -- place `size` floats on the stack (free).
- `tracker.read(name, size)` -- observe stack positions, accumulate DMD. Returns list of per-element distances.
- `tracker.summary()` -- returns dict with `dmd`, `reads`, `writes`, `stack_size`, `per_buffer`.
- `tracker.report()` -- prints formatted report.

## Limitations

- **Performance overhead.** O(n) per element access where n = stack size. GF(2) with n=20 takes ~24s under tracking. For measurement, not production.
- **Array-level granularity.** Matmul records one bulk read of each input, not the per-element access pattern of the inner loop.
- **Pure-python loops.** Scalar access like `aug[row, col] == 1` generates per-element tracking events.
- **Non-numpy code.** Values extracted as plain Python scalars leave the tracking world.
- **Constructor coverage.** Only `np.zeros`, `np.ones`, `np.empty` are patched. `np.arange`, `np.linspace`, etc. return plain arrays.

## Files

| File | What it is |
|------|-----------|
| `src/sparse_parity/lru_tracker.py` | LRUStackTracker |
| `src/sparse_parity/tracked_numpy.py` | TrackedArray + tracking_context |
| `tests/test_tracked_numpy.py` | 30 tests |

# TrackedArray: Auto-instrumented DMC Tracking

## Problem

The existing MemTracker requires manual `tracker.read()` / `tracker.write()` calls placed throughout experiment code. This is error-prone: the GF(2) experiment reported DMC of 8,607 but the actual Gaussian elimination (row swaps, XOR operations) had zero tracking calls. The real DMC is ~189K-227K.

Manual instrumentation also creates a barrier for competition submissions. If someone submits a new algorithm, we can't trust them to instrument correctly, and we shouldn't require them to.

## Solution

`TrackedArray` is an `np.ndarray` subclass that automatically records every read and write on a MemTracker. Wrap the inputs, run unmodified numpy code, read the DMC at the end.

```python
from sparse_parity.tracked_numpy import TrackedArray, tracking_context
from sparse_parity.tracker import MemTracker

tracker = MemTracker()
with tracking_context(tracker):
    A = TrackedArray(A_raw, "A", tracker)
    b = TrackedArray(b_raw, "b", tracker)
    solution, rank = gf2_gauss_elim(A.copy(), b.copy())
print(tracker.summary()["dmc"])  # 227,508
```

Zero changes inside `gf2_gauss_elim`.

## How it works

### TrackedArray (ndarray subclass)

Every TrackedArray carries a reference to a MemTracker and a buffer name. Operations are intercepted at three levels:

1. **`__array_ufunc__`** -- catches all ufuncs: `+`, `-`, `*`, `^`, `==`, `<`, etc. For each TrackedArray input, records a `tracker.read()`. For the output, creates a new TrackedArray and records a `tracker.write()`. In-place ops (with `out=`) write back to the existing buffer name.

2. **`__array_function__`** -- catches numpy functions: `np.where`, `np.prod`, `np.sum`, `np.all`, `np.sort`, `np.concatenate`, etc. Same read/write pattern. A default handler covers any unregistered function. Specific functions can be registered with `@implements(np.func)` for custom behavior.

3. **`__getitem__` / `__setitem__`** -- catches indexing, slicing, and fancy indexing. `__getitem__` records a read sized to the actual slice (not the whole array), then wraps the result as a new TrackedArray. `__setitem__` records a read of the source and a write to the target slice.

### tracking_context (context manager)

The context manager solves a bootstrapping problem: numpy constructor functions like `np.zeros(shape)` take no array arguments, so `__array_function__` never fires for them. Inside a `tracking_context`, these constructors are monkey-patched to return TrackedArrays. The patches are reverted when the context exits.

Patched constructors: `np.zeros`, `np.ones`, `np.empty`.

### Propagation

Tracking propagates automatically. Any array derived from a TrackedArray (via ufunc, slice, function call) is itself a TrackedArray on the same tracker. This means wrapping the initial inputs is sufficient; all intermediate results inherit tracking.

### Buffer naming

Each new buffer gets an auto-generated name like `_bitwise_xor_42` or `_slice_15`. The counter resets between experiments via `reset_counter()`. User-provided names (like `"A_gf2"`) are used for the initial wraps.

## Accounting model

Each operation is modeled as:
- **Reads**: every input TrackedArray is read. Size = the array's `.size` for ufuncs, or the slice size for indexing.
- **Writes**: every output is a new buffer (or overwrites an existing one for in-place ops). Size = the output's `.size`.

This matches the MemTracker's existing clock model: clock advances by `size` floats per read or write, and reuse distance = clock difference between write and subsequent read of the same buffer.

### Comparison with manual tracking

| Method | DMC (n=20, k=3) | Notes |
|--------|-----------------|-------|
| Manual harness (I/O only) | 8,607 | Only tracks data conversion, misses elimination |
| Yad's honest estimate | 189,056 | Manual count of row operations |
| TrackedArray auto | 227,508 | Includes all intermediates (slices, XOR results) |

The auto-tracked number is ~20% higher than the honest estimate because each intermediate result (row slice, XOR output) is a separate buffer with its own write+read cycle. This overhead is consistent and can be considered a more conservative (higher-fidelity) measurement.

## Limitations

- **Overhead**: TrackedArray adds Python-level overhead per operation. Not suitable for performance-critical training loops. Use it for measurement, not production.
- **Pure-python loops**: The GF(2) pivot search does `aug[row, col] == 1` in a Python for-loop. Each scalar access is tracked individually. This is correct but generates many events.
- **Non-numpy code**: Operations that extract plain python values (e.g., `int(arr[0])`) exit the tracking world. The read is captured but any downstream computation on the plain value is not.
- **Constructor coverage**: Only `np.zeros`, `np.ones`, `np.empty` are patched. Other constructors like `np.arange`, `np.linspace`, `np.random.*` return plain arrays unless inside a tracked operation.

## Files

- `src/sparse_parity/tracked_numpy.py` -- TrackedArray implementation
- `tests/test_tracked_numpy.py` -- 19 tests covering ufuncs, indexing, context manager, and GF(2) integration
- `src/sparse_parity/tracker.py` -- existing MemTracker (unchanged)

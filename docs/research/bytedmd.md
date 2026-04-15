# ByteDMD (Primary Metric)

ByteDMD is the new primary energy-cost metric for sparse parity submissions. Pure Python, byte-granularity, deterministic. Source of truth: [cybertronai/ByteDMD](https://github.com/cybertronai/ByteDMD). Vendored locally at `src/bytedmd/`.

## Why it replaces TrackedArray

TrackedArray hooks numpy operations. It works, but values can escape into numpy's C extensions (e.g. `np.asarray()`, bit-packed ints), and it tracks at element granularity which doesn't reflect the cost difference between a uint8 and a float64.

ByteDMD operates on Python values directly. It traces Python-level operations via dunder methods. Byte-level granularity rewards smaller dtypes. No escape hatches as long as the submission stays in pure Python.

TrackedArray is kept as the legacy tracker for existing experiments. It is no longer the metric for new challenge submissions.

## The metric

Every value lives in an LRU stack measured in bytes. Reading a value at depth `d` costs `ceil(sqrt(d))` per byte. Writes are free -- they just push to the top. Existing entries get compacted when their last use is reached.

Three rules:

1. **Simultaneous pricing**: all inputs to an instruction are priced against the pre-instruction stack state, not sequentially. Guarantees `a + b == b + a` cost-wise.
2. **Eager initialization**: arguments load left to right. First argument sits at depth 1.
3. **Aggressive compaction**: dead values are dropped immediately. Stack stays small.

## Quick start

```python
from bytedmd import bytedmd, traced_eval

def my_add(a, b, c):
    return (a + b) + c

cost = bytedmd(my_add, (1, 2, 3))         # 6
trace, result = traced_eval(my_add, (1, 2, 3))  # ([1, 2, 1, 2], 6)
```

The `bytedmd()` function returns total cost. `traced_eval()` returns the per-read trace plus the function's return value. Trace entries are stack depths at each read.

For multi-byte values (e.g. 32-bit ints):

```python
cost = bytedmd(my_add, (1, 2, 3), bytes_per_element=4)
```

## What works, what doesn't

Tracked: arithmetic (`+ - * / // %`), comparisons (`< > == !=`), bitwise (`& | ^ << >>`), boolean (`and or not`), indexing (`a[i]`), conditionals on tracked values.

Not tracked: numpy operations (anything that calls into C), pure-memory copies (compaction handles them), constant operations.

See `tests/test_bytedmd_gotchas.py` for known edge cases.

## Files

| File | What |
|------|------|
| `src/bytedmd/__init__.py` | Vendored implementation |
| `src/bytedmd/README.md` | Vendored README |
| `tests/test_bytedmd.py` | 10 unit tests |
| `tests/test_bytedmd_gotchas.py` | 3 edge case tests |

## References

- Reference repo: https://github.com/cybertronai/ByteDMD
- Original DMD paper: Ding et al., arXiv:2312.14441
- Bill Dally on memory cost models: https://cacm.acm.org/opinion/on-the-model-of-computation-point/
- Wesley Smith meeting notes (where byte-granularity was decided): in `docs/google-docs/` after sync

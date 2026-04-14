> **Vendored copy.** Source of truth: <https://github.com/cybertronai/ByteDMD>. Some links in this file (to `docs/`, `benchmarks/`, etc.) point to paths that exist in the upstream repo but not here. Refer to the upstream for the full documentation, benchmarks, and figures.

# A cost model of complexity for the 21st century: ByteDMD

Data movement matters more than FLOPs. Recently accessed bytes can be cached, penalize non-local reads using the following cost model:

$$C=\sum_{b \in bytes} \sqrt{D(b)}$$

where $D(b)$ is the depth of byte $b$ in the LRU stack. Square-root is motivated by VLSI routing cost in 2D.

## Usage

```python
from bytedmd import bytedmd

def dot(a, b):
    return sum(i1*i2 for (i1,i2) in zip(a,b))

a = [0, 1]
b = [2, 3]

# dot product
assert dot(a,b) == 3

# ByteDMD cost of dot product
assert bytedmd(dot, (a, b)) == 12
```

**Interactive visualization:** [dot product on the LRU stack](https://yaroslavvb.github.io/ByteDMD-vis/dotproduct_stack.html) — animated walkthrough of how the stack evolves during a dot product computation.

## Motivation


Modern architectures spend more energy moving data than doing arithmetic, making FLOP counts an outdated cost metric. Bill Dally ([ACM Opinion](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)) proposed penalizing data movement based on 2D spatial distance to the processor. To avoid manual spatial mapping, Ding and Smith ([Beyond Time Complexity, 2022](https://arxiv.org/abs/2203.02536)) automated this via Data Movement Distance (DMD): a rule treating memory as an LRU stack where reading a byte at depth $d$ costs $\sqrt{d}$, modeling a cache laid out in 2D.

To avoid floating point issues, we round up to the nearest integer.

![ByteDMD](docs/ceil_figure.svg)

This rounding corresponds to routing wire length on a 2D grid with LRU stack arranged in the following order.

![ByteDMD](docs/manhattan_figure.svg)

## Computation Model

An idealized processor operates directly on an element-level LRU stack. **Computations and writes are free; only memory reads incur a cost.**

- **Stack State:** Ordered from least recently used (bottom) to most recently used (top). Depth is measured in bytes from the top (topmost byte = depth 1). Multi-byte scalars are treated as contiguous blocks of bytes.
- **Eager initialization:** Arguments are loaded onto the stack left to right — the first argument sits at the top (depth 1). All input elements are live and addressable from the start.
- **Read Cost:** Reading a byte at depth $d$ costs $\lceil\sqrt{d}\rceil$.
- **Simultaneous pricing:** All inputs to an instruction are priced against the stack state *before* any LRU bumping. This guarantees commutativity: `Cost(a+b) == Cost(b+a)`.
- **Only live contribute to depth of the stack:** Any value that's dead (no longer used) is immediately removed from the stack and remaining elements slide up to close the gap. This models an optimal compiler that keeps the stack clamped to the active working set.

### Instruction Semantics

See [Instruction Set](docs/instruction_set.md) for the complete list of supported instructions.

For an instruction with inputs $x_1, \dots, x_m$ and outputs $y_1, \dots, y_n$ with $m\ge 1, n\ge 0$

1. **Price reads:** Evaluate $\sum C(x_j)$ simultaneously against the stack state *before* the instruction begins. All inputs see the same pre-instruction snapshot. Repeated inputs are charged per occurrence at the same depth (e.g., `a + a` charges `⌈√d⌉` twice where `d` is `a`'s pre-instruction depth).
2. **Update LRU:** Batch-move unique inputs to the top of the stack in read order. `b + c` and `c + b` yield the same cost (commutativity) but may differ in final stack order.
3. **Push outputs:** Allocate new output blocks and push them to the top at zero cost.

## Example Walkthrough

Consider the following function with three scalar arguments:

```python
def my_add(a, b, c):
    return (a + b) + c
```

**1. Initial Stack (left = top, right = bottom)** 
Arguments are loaded left to right — first argument at the top:
```text
[a, b, c]    ← a at distance 1, b at distance 2, c at distance 3
```

**2. First operation: `a + b`**  
Both operands are priced simultaneously against the initial stack:

$$C(a) + C(b) = \lceil\sqrt{1}\rceil + \lceil\sqrt{2}\rceil = 1 + 2 = 3$$

After LRU bumping and pushing the result `t = a + b`:
```text
[t, b, a, c]    ← t at distance 1, b at distance 2, a at distance 3, c at distance 4
```
Liveness analysis evicts `a` and `b` (their last use just happened):
```text
[t, c]    ← t at distance 1, c at distance 2
```

**3. Second operation: `t + c`**  
$$C(t) + C(c) = \lceil\sqrt{1}\rceil + \lceil\sqrt{2}\rceil = 1 + 2 = 3$$

**Total cost:** $3 + 3 = 6$. Trace: `[1, 2, 1, 2]`.


## Inspecting the IR

The tracer also emits a small **intermediate representation** that makes the
LRU stack lifecycle explicit. Three event types: `STORE k` (allocate vk on
top), `READ k@d` (read vk at depth d and LRU-bump), `OP name(vk@d, …)`
(summary of the preceding reads — this is what incurs cost). Op results are
materialized by the `STORE` that immediately follows the `OP`.

```python
from bytedmd import inspect_ir, format_ir, bytedmd

def matvec2(A, x):
    y0 = A[0][0]*x[0] + A[0][1]*x[1]
    y1 = A[1][0]*x[0] + A[1][1]*x[1]
    return [y0, y1]

print(format_ir(inspect_ir(matvec2, ([[1,2],[3,4]], [5,6]))))
```

```text
STORE v1                                # x[0] loaded first (deepest)
STORE v2                                # x[1]
STORE v3                                # A[0][0]
STORE v4                                # A[0][1]
STORE v5                                # A[1][0]
STORE v6                                # A[1][1]
  READ v3@4  cost=2                     # A[0][0] (left-to-right: A at top)
  READ v1@6  cost=3                     # x[0] at bottom
OP    mul(v3@4, v1@6)  cost=5           # A[0][0]*x[0]
STORE v7
  READ v4@5  cost=3                     # A[0][1] (v3 evicted after last use)
  READ v2@6  cost=3                     # x[1]
OP    mul(v4@5, v2@6)  cost=6           # A[0][1]*x[1]
STORE v8
  READ v7@3  cost=2                     # hot hit: v7 sank as v4, v2 entered
  READ v8@1  cost=1                     # hot hit: v8 still at top
OP    add(v7@3, v8@1)  cost=3           # y0
STORE v9
  READ v5@5  cost=3                     # A[1][0] (dead temps evicted)
  READ v1@3  cost=2                     # hot hit: x[0] still on stack
OP    mul(v5@5, v1@3)  cost=5
STORE v10
  READ v6@4  cost=2                     # A[1][1]
  READ v2@3  cost=2                     # hot hit: x[1]
OP    mul(v6@4, v2@3)  cost=4
STORE v11
  READ v10@2  cost=2
  READ v11@1  cost=1
OP    add(v10@2, v11@1)  cost=3         # y1
STORE v12
# total cost = 26
```

Note the left-to-right initialization: `A` elements (the first argument) sit
at the top of the stack, while `x` elements (the second argument) are
deeper. Liveness analysis aggressively evicts dead variables: after `v3`'s
single read, it is removed and remaining elements slide up. This keeps the
stack clamped to the active working set.

## ByteDMD benchmarks

See "benchmarks/" folder

### Linear algebra (`benchmark_linalg.py`)

ByteDMD cost by method and matrix size `N`:

```
|    |  matvec |   vecmat | naive matmul |
|  N | (y=A@x) | (y=xᵀ@A) |      (i-j-k) |
|----|---------|----------|--------------|
|  2 |      26 |       25 |           57 |
|  4 |     157 |      150 |          720 |
|  8 |     896 |      832 |        8,867 |
| 16 |   5,354 |    4,688 |      109,783 |
```

### microGPT single-token forward pass

Architecture: `vocab=4, embd=4, heads=2, head_dim=2, 1 layer, block_size=4`.
Based on [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| microGPT (1 layer, embd=4) | single token forward | 3214 |

# Reports

In-depth reports applying ByteDMD to specific algorithms and design questions:

- [Strassen vs naive matmul](docs/report-strassen-benchmarks/report.md) — at what matrix size does Strassen's recursive algorithm beat naive matmul under ByteDMD? Includes a crossover-point experiment.
- [Modern flash attention vs naive attention](docs/report-modern-flash-attention/report.md) — full sweep across sequence length, head dim, and block size showing flash attention's advantage growing as O(sqrt(N/Bk)) under ByteDMD while FLOPs see no benefit. Uses an optimised tracer (`bytedmd_fast.py`).
- [Antigravity flash attention experiments](docs/report-antigravity-flash-attention/report.md) — alternative flash attention implementations and their ByteDMD costs.
- [Attention benchmark notes](benchmarks/attention_report.md) — the small-scale flash vs naive results that motivated the modern-attention deep dive.

# Python Gotcha's
The tracer implements ByteDMD by wrapping Python objects. This means that the "Instruction Set" of this metric corresponds to Python built-ins, documented under [docs/instruction_set.md](docs/instruction_set.md).

Python behavior means this implementation occasionally doesn't match README semantics and it is possible to escape the wrapping mechanism (local arrays, exception side-channels, identity ops, type introspection, f-strings, math.trunc/ceil/floor on tracked values, etc.). Known failure cases are documented in `test_gotchas.py` — avoid those patterns when writing code you want measured.


[Original Google Doc](https://docs.google.com/document/d/1sj5NqOg6Yqh10bXzGVEF5uIzSjFWAnqqTE75AMng2-s/edit?tab=t.0#heading=h.ujy6ygk7sjmb)


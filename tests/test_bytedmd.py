#!/usr/bin/env uv run python
import numpy as np
from bytedmd import bytedmd, traced_eval, trace_to_bytedmd


def my_add(a, b, c):
    return (a + b) + c


def test_my_add():
    cost = bytedmd(my_add, (1, 2, 3))
    assert cost == 6

    # Stack starts as [c, b, a] (a at top, depth 1). a+b reads a@1, b@2.
    # After compaction: [c, t]. t+c reads t@1, c@2. Total: 6.
    trace, _ = traced_eval(my_add, (1, 2, 3))
    assert trace == [1, 2, 1, 2]

    assert trace_to_bytedmd(trace, bytes_per_element=1) == 6
    assert trace_to_bytedmd(trace, bytes_per_element=2) == 14

    assert bytedmd(my_add, (1, 2, 3), bytes_per_element=2) == 14


def my_composite_func(a, b, c, d):
    e = b + c
    f = a + d
    return e > f


def test_repeated_operand_is_charged_twice():
    """README says a+a should charge two reads against the same pre-instruction stack."""
    trace, _ = traced_eval(lambda a: a + a, (5,))
    assert trace == [1, 1]


def test_my_composite_func():
    trace, result = traced_eval(my_composite_func, (1, 2, 3, 4))
    assert trace == [2, 3, 2, 3, 2, 1]
    cost = bytedmd(my_composite_func, (1, 2, 3, 4))
    assert cost == 11

def test_dot_product():
    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1]

    a, b = [0, 1], [2, 3]
    trace, result = traced_eval(dot, (a, b))

    assert trace == [2, 4, 2, 3, 2, 1]
    assert result == 3
    assert bytedmd(dot, (a, b)) == 11


def test_branching_and_comparisons_trace():
    def my_relu(a):
        if a > 0:
            return a * 2
        return a
        
    # Branch taken: a > 0 reads a, __bool__ reads result, then a * 2 reads a.
    # Under aggressive compaction, the comparison result is evicted after
    # __bool__ (its last use), so the later read of `a` sees depth 1.
    trace_pos, _ = traced_eval(my_relu, (5,))
    assert trace_pos == [1, 1, 1]

    # Branch skipped: a > 0 reads a, __bool__ reads result
    trace_neg, _ = traced_eval(my_relu, (-5,))
    assert trace_neg == [1, 1]


def test_divmod_tuple_allocation_trace():
    """
    6. Tests operations natively returning multiple tracked values.
    divmod(a, b) evaluates to a tuple (q, r), sequentially triggering 
    multiple allocations on the LRU stack.
    """
    def my_divmod(a, b):
        q, r = divmod(a, b)
        return q + r + a
        
    trace, result = traced_eval(my_divmod, (10, 3))
    assert trace == [1, 2, 2, 1, 1, 2]


def test_implicit_boolean_is_traced():
    """
    `if a:` now correctly calls __bool__, recording a read and evaluating truthiness properly.
    Both implicit and explicit branches produce the same result for a=0.
    """
    def implicit_branch(a):
        if a:
            return a + 10
        return a

    def explicit_branch(a):
        if a != 0:
            return a + 10
        return a

    trace_implicit, result_implicit = traced_eval(implicit_branch, (0,))
    # __bool__ reads a, then takes the else branch returning a (no further read)
    assert trace_implicit == [1]
    assert result_implicit == 0

    trace_explicit, result_explicit = traced_eval(explicit_branch, (0,))
    # a != 0 reads a, __bool__ reads the comparison result
    assert trace_explicit == [1, 1]
    assert result_explicit == 0


def test_index_protocol_works():
    trace, result = traced_eval(lambda n: [i for i in range(n)], (3,))
    assert trace == [1]
    assert result == [0, 1, 2]

    trace, result = traced_eval(lambda xs, i: xs[i], ([10, 20, 30], 1))
    assert trace == [2]
    assert result == 20


def test_not_is_traced():
    """
    `not a` now invokes __bool__, generating a read trace and returning the correct result.
    """
    trace, result = traced_eval(lambda a: not a, (0,))
    assert trace == [1]
    assert result is True


def _matvec(A, x):
    n = len(x)
    y = [None] * n
    for i in range(n):
        s = A[i][0] * x[0]
        for j in range(1, n):
            s = s + A[i][j] * x[j]
        y[i] = s
    return y


def _vecmat(A, x):
    n = len(x)
    y = [None] * n
    for j in range(n):
        s = x[0] * A[0][j]
        for i in range(1, n):
            s = s + x[i] * A[i][j]
        y[j] = s
    return y


def _ceil_sqrt(x):
    """ceil(sqrt(x)) via integer arithmetic."""
    import math
    return math.isqrt(x - 1) + 1 if x > 0 else 0


def test_matvec_costs():
    """Verify matvec costs with eager init + aggressive compaction.
    With eager init, matvec and vecmat are no longer symmetric because
    the traversal order matters against the pre-loaded stack."""
    expected_mv = {2: 26, 3: 75, 4: 157, 5: 270, 6: 422, 7: 615, 8: 896}
    expected_vm = {2: 25, 3: 72, 4: 150, 5: 258, 6: 397, 7: 572, 8: 832}
    for n in [2, 3, 4, 5, 6, 7, 8]:
        A = np.ones((n, n))
        x = np.ones(n)
        mv = bytedmd(_matvec, (A, x))
        vm = bytedmd(_vecmat, (A, x))
        assert mv == expected_mv[n], f"matvec N={n}: got {mv}, expected {expected_mv[n]}"
        assert vm == expected_vm[n], f"vecmat N={n}: got {vm}, expected {expected_vm[n]}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

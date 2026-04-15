#!/usr/bin/env python3
"""
Python gotchas: Because we implement ByteDMD in Python by wrapping Python objects,
our Python framework deviates from idealized description in the README
for certain cases, illustrated by tests below.

Future release of ByteDMD metric may fix this to be more faithful to the README.md
"""
from bytedmd import traced_eval


def test_gotcha_constant_ops():
    """
    Limitation of Python model: constants are not tracked.
    """
    def f(a):
        return 10 - a * 10

    trace, _ = traced_eval(f, (5,))
    assert trace == [1, 1]


def test_gotcha_pure_memory_movement_is_free():
    """
    Limitation of Python model: pure list index without computation does not trigger
    math magic methods, hence generating no read trace.
    """
    def transpose(A):
        n = len(A)
        return [[A[j][i] for j in range(n)] for i in range(n)]

    A = [[1, 2], [3, 4]]
    trace, result = traced_eval(transpose, (A,))

    assert trace == []
    assert result == [[1, 3], [2, 4]]


def test_short_circuit_gotcha():
    """
    Python short-circuit means only one operand may be traced.
    With eager init, both a and b are pre-loaded on the stack.

    `a and b` with a=0: only a is read. b is never used and gets
    compacted away, so a sits at depth 1.

    `a or b` with a=0: a.__bool__() reads a. With left-to-right init,
    a is at the top (depth 1).
    """
    def logical_and(a, b):
        return a and b

    trace, result = traced_eval(logical_and, (0, 5))
    assert trace == [1]
    assert result == 0

    trace, result = traced_eval(lambda a, b: a or b, (0, 5))
    assert trace == [1]
    assert result == 5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

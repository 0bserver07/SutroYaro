"""Auto-instrumented numpy wrapper for DMD tracking.

Wraps numpy arrays so that every operation (ufuncs, indexing, slicing,
numpy functions) automatically records reads and writes on a tracker.
Works with both LRUStackTracker (recommended) and MemTracker (legacy).

Usage:
    from sparse_parity.tracked_numpy import TrackedArray, tracking_context
    from sparse_parity.lru_tracker import LRUStackTracker

    tracker = LRUStackTracker()
    with tracking_context(tracker):
        A = TrackedArray(A_raw, "A", tracker)
        b = TrackedArray(b_raw, "b", tracker)
        # Run unmodified numpy code -- all ops auto-tracked
        solution, rank = gf2_gauss_elim(A.copy(), b.copy())
    print(tracker.summary()["read_dmd"])
"""

import numpy as np
import threading
from contextlib import contextmanager

_next_id = 0
_local = threading.local()


def _auto_name(prefix="arr"):
    """Generate a unique buffer name for unnamed intermediate arrays."""
    global _next_id
    _next_id += 1
    return f"_{prefix}_{_next_id}"


def reset_counter():
    """Reset the auto-name counter (useful between experiments)."""
    global _next_id
    _next_id = 0


def get_active_tracker():
    """Get the tracker set by the nearest enclosing tracking_context, or None."""
    return getattr(_local, 'tracker', None)


@contextmanager
def tracking_context(tracker):
    """Set a thread-local active tracker so np.zeros etc. produce TrackedArrays.

    Also monkey-patches numpy constructor functions (zeros, ones, empty, etc.)
    so that arrays created inside the block are automatically tracked.
    """
    old = getattr(_local, 'tracker', None)
    _local.tracker = tracker

    # Patch numpy constructors that don't take array args
    # (so __array_function__ never fires for them)
    _orig_zeros = np.zeros
    _orig_ones = np.ones
    _orig_empty = np.empty

    def _patched_zeros(shape, dtype=float, order='C', **kw):
        result = _orig_zeros(shape, dtype=dtype, order=order, **kw)
        name = _auto_name("zeros")
        out = result.view(TrackedArray)
        out._tracker = tracker
        out._buf_name = name
        tracker.write(name, result.size)
        return out

    def _patched_ones(shape, dtype=float, order='C', **kw):
        result = _orig_ones(shape, dtype=dtype, order=order, **kw)
        name = _auto_name("ones")
        out = result.view(TrackedArray)
        out._tracker = tracker
        out._buf_name = name
        tracker.write(name, result.size)
        return out

    def _patched_empty(shape, dtype=float, order='C', **kw):
        result = _orig_empty(shape, dtype=dtype, order=order, **kw)
        name = _auto_name("empty")
        out = result.view(TrackedArray)
        out._tracker = tracker
        out._buf_name = name
        tracker.write(name, result.size)
        return out

    np.zeros = _patched_zeros
    np.ones = _patched_ones
    np.empty = _patched_empty

    try:
        yield tracker
    finally:
        _local.tracker = old
        np.zeros = _orig_zeros
        np.ones = _orig_ones
        np.empty = _orig_empty


class TrackedArray(np.ndarray):
    """ndarray subclass that auto-records reads/writes on a tracker.

    Works with any tracker that has write(name, size) and read(name, size)
    methods (LRUStackTracker or MemTracker).

    Every operation that reads this array records a tracker.read().
    Every operation that produces a new array records a tracker.write().
    Tracking propagates: derived arrays are also TrackedArrays.
    """

    def __new__(cls, array, name=None, tracker=None):
        obj = np.asarray(array).view(cls)
        obj._tracker = tracker
        obj._buf_name = name or _auto_name()
        if tracker is not None:
            tracker.write(obj._buf_name, obj.size)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._tracker = getattr(obj, '_tracker', None)
        self._buf_name = getattr(obj, '_buf_name', None)

    def _record_read(self):
        """Record that this array was read."""
        if self._tracker is not None:
            self._tracker.read(self._buf_name, self.size)

    def _make_tracked(self, result, prefix="result"):
        """Wrap a result array as TrackedArray with a fresh name."""
        if not isinstance(result, np.ndarray):
            return result
        if self._tracker is None:
            return result
        name = _auto_name(prefix)
        out = result.view(TrackedArray)
        out._tracker = self._tracker
        out._buf_name = name
        self._tracker.write(name, result.size)
        return out

    # --- ufunc interception (handles +, -, *, ^, ==, <, >, etc.) ---

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Record reads for all TrackedArray inputs
        tracker = None
        for inp in inputs:
            if isinstance(inp, TrackedArray) and inp._tracker is not None:
                inp._record_read()
                if tracker is None:
                    tracker = inp._tracker

        # Also handle `out` kwarg (in-place operations like np.add(a, b, out=c))
        out_args = kwargs.get('out', ())
        for o in out_args:
            if isinstance(o, TrackedArray) and o._tracker is not None:
                tracker = o._tracker

        # Cast inputs to plain ndarray so numpy doesn't recurse
        plain_inputs = tuple(
            np.asarray(x) if isinstance(x, TrackedArray) else x
            for x in inputs
        )
        plain_kwargs = dict(kwargs)
        if out_args:
            plain_kwargs['out'] = tuple(
                np.asarray(x) if isinstance(x, TrackedArray) else x
                for x in out_args
            )

        result = getattr(ufunc, method)(*plain_inputs, **plain_kwargs)

        if tracker is None:
            return result

        # If output was written in-place, record the write on existing buffer
        if out_args:
            for o_orig, o_plain in zip(out_args, plain_kwargs['out']):
                if isinstance(o_orig, TrackedArray) and o_orig._tracker is not None:
                    o_orig._tracker.write(o_orig._buf_name, o_orig.size)
            # Return the original TrackedArray for in-place ops
            if len(out_args) == 1:
                return out_args[0]
            return out_args

        # Wrap result
        if isinstance(result, np.ndarray):
            name = _auto_name(ufunc.__name__)
            out = result.view(TrackedArray)
            out._tracker = tracker
            out._buf_name = name
            tracker.write(name, result.size)
            return out
        elif isinstance(result, tuple):
            return tuple(
                self._make_tracked(r, ufunc.__name__) if isinstance(r, np.ndarray) else r
                for r in result
            )
        return result

    # --- numpy function interception (np.where, np.prod, np.sum, etc.) ---

    HANDLED_FUNCTIONS = {}

    def __array_function__(self, func, types, args, kwargs):
        if func in TrackedArray.HANDLED_FUNCTIONS:
            return TrackedArray.HANDLED_FUNCTIONS[func](*args, **kwargs)
        return _default_array_function(func, args, kwargs)

    # --- Indexing ---

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if self._tracker is not None:
            # Track the size of what was actually read, not the whole array
            read_size = result.size if isinstance(result, np.ndarray) else 1
            self._tracker.read(self._buf_name, read_size)
            if isinstance(result, np.ndarray):
                name = _auto_name("slice")
                out = result.view(TrackedArray)
                out._tracker = self._tracker
                out._buf_name = name
                self._tracker.write(name, result.size)
                return out
        return result

    def __setitem__(self, key, value):
        # Read the value being assigned
        if isinstance(value, TrackedArray):
            value._record_read()
        # Write to this buffer (the target)
        super().__setitem__(key, value)
        if self._tracker is not None:
            # For row/element assignment, track the size of the written slice
            try:
                target = np.asarray(self)[key]
                write_size = target.size if isinstance(target, np.ndarray) else 1
            except (IndexError, TypeError):
                write_size = self.size
            self._tracker.write(self._buf_name, write_size)

    # --- Common methods that should track ---

    def copy(self, order='C'):
        self._record_read()
        result = np.array(self, order=order, copy=True)
        return self._make_tracked(result, "copy")

    def astype(self, dtype, **kwargs):
        self._record_read()
        result = super().astype(dtype, **kwargs)
        return self._make_tracked(np.asarray(result), "astype")

    def sum(self, axis=None, **kwargs):
        self._record_read()
        result = np.asarray(self).sum(axis=axis, **kwargs)
        if isinstance(result, np.ndarray) and self._tracker is not None:
            return self._make_tracked(result, "sum")
        return result

    def tolist(self):
        self._record_read()
        return np.asarray(self).tolist()

    @property
    def T(self):
        # Transpose is a zero-cost view in numpy (no data movement).
        # Return a TrackedArray sharing the same buffer name.
        result = super().T
        if isinstance(result, np.ndarray) and self._tracker is not None:
            out = result.view(TrackedArray)
            out._tracker = self._tracker
            out._buf_name = self._buf_name  # same buffer, just a view
            return out
        return result


# --- Register numpy function implementations ---

def implements(np_function):
    """Decorator to register a TrackedArray implementation of a numpy function."""
    def decorator(func):
        TrackedArray.HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


def _find_tracker(*args):
    """Find the tracker from any TrackedArray in args."""
    for a in args:
        if isinstance(a, TrackedArray) and a._tracker is not None:
            return a._tracker
    return None


def _record_reads(*args):
    """Record reads for all TrackedArray args."""
    for a in args:
        if isinstance(a, TrackedArray):
            a._record_read()


def _strip_tracked(arg):
    """Recursively strip TrackedArrays from an arg, recording reads. Returns (plain, tracker)."""
    if isinstance(arg, TrackedArray):
        arg._record_read()
        return np.asarray(arg), arg._tracker
    elif isinstance(arg, (list, tuple)):
        tracker = None
        plain = []
        for item in arg:
            p, t = _strip_tracked(item)
            plain.append(p)
            if t is not None:
                tracker = t
        return type(arg)(plain), tracker
    return arg, None


def _default_array_function(func, args, kwargs):
    """Fallback: record reads, call numpy, wrap output."""
    tracker = None
    plain_args = []
    for a in args:
        p, t = _strip_tracked(a)
        plain_args.append(p)
        if t is not None:
            tracker = t

    plain_kwargs = {}
    for k, v in kwargs.items():
        p, t = _strip_tracked(v)
        plain_kwargs[k] = p
        if t is not None:
            tracker = t

    result = func(*plain_args, **plain_kwargs)

    if tracker is not None and isinstance(result, np.ndarray):
        name = _auto_name(func.__name__)
        out = result.view(TrackedArray)
        out._tracker = tracker
        out._buf_name = name
        tracker.write(name, result.size)
        return out
    return result


@implements(np.zeros)
def tracked_zeros(shape, dtype=float, order='C'):
    result = np.zeros(shape, dtype=dtype, order=order)
    tracker = get_active_tracker()
    if tracker is not None:
        name = _auto_name("zeros")
        out = result.view(TrackedArray)
        out._tracker = tracker
        out._buf_name = name
        tracker.write(name, result.size)
        return out
    return result


@implements(np.zeros_like)
def tracked_zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    tracker = a._tracker if isinstance(a, TrackedArray) else None
    result = np.zeros_like(np.asarray(a), dtype=dtype, order=order, subok=False, shape=shape)
    if tracker is not None:
        name = _auto_name("zeros_like")
        out = result.view(TrackedArray)
        out._tracker = tracker
        out._buf_name = name
        tracker.write(name, result.size)
        return out
    return result


@implements(np.where)
def tracked_where(condition, x=None, y=None):
    tracker = _find_tracker(condition, x, y) if x is not None else _find_tracker(condition)
    _record_reads(condition)
    if x is not None:
        _record_reads(x, y)

    cond_plain = np.asarray(condition)
    if x is None:
        return np.where(cond_plain)

    x_plain = np.asarray(x) if isinstance(x, np.ndarray) else x
    y_plain = np.asarray(y) if isinstance(y, np.ndarray) else y
    result = np.where(cond_plain, x_plain, y_plain)

    if tracker is not None and isinstance(result, np.ndarray):
        name = _auto_name("where")
        out = result.view(TrackedArray)
        out._tracker = tracker
        out._buf_name = name
        tracker.write(name, result.size)
        return out
    return result


@implements(np.prod)
def tracked_prod(a, axis=None, **kwargs):
    if isinstance(a, TrackedArray):
        a._record_read()
        tracker = a._tracker
    else:
        tracker = None
    result = np.prod(np.asarray(a), axis=axis, **kwargs)
    if tracker is not None and isinstance(result, np.ndarray):
        name = _auto_name("prod")
        out = result.view(TrackedArray)
        out._tracker = tracker
        out._buf_name = name
        tracker.write(name, result.size)
        return out
    return result


@implements(np.sum)
def tracked_sum(a, axis=None, **kwargs):
    if isinstance(a, TrackedArray):
        a._record_read()
        tracker = a._tracker
    else:
        tracker = None
    result = np.sum(np.asarray(a), axis=axis, **kwargs)
    if tracker is not None and isinstance(result, np.ndarray):
        name = _auto_name("sum")
        out = result.view(TrackedArray)
        out._tracker = tracker
        out._buf_name = name
        tracker.write(name, result.size)
        return out
    return result


@implements(np.mean)
def tracked_mean(a, axis=None, **kwargs):
    if isinstance(a, TrackedArray):
        a._record_read()
        tracker = a._tracker
    else:
        tracker = None
    result = np.mean(np.asarray(a), axis=axis, **kwargs)
    if tracker is not None and isinstance(result, np.ndarray):
        name = _auto_name("mean")
        out = result.view(TrackedArray)
        out._tracker = tracker
        out._buf_name = name
        tracker.write(name, result.size)
        return out
    return result


@implements(np.all)
def tracked_all(a, axis=None, **kwargs):
    if isinstance(a, TrackedArray):
        a._record_read()
    return np.all(np.asarray(a), axis=axis, **kwargs)


@implements(np.sort)
def tracked_sort(a, axis=-1, **kwargs):
    if isinstance(a, TrackedArray):
        a._record_read()
        tracker = a._tracker
    else:
        tracker = None
    result = np.sort(np.asarray(a), axis=axis, **kwargs)
    if tracker is not None:
        name = _auto_name("sort")
        out = result.view(TrackedArray)
        out._tracker = tracker
        out._buf_name = name
        tracker.write(name, result.size)
        return out
    return result


@implements(np.concatenate)
def tracked_concatenate(arrays, axis=0, **kwargs):
    tracker = None
    plain = []
    for a in arrays:
        if isinstance(a, TrackedArray):
            a._record_read()
            if tracker is None:
                tracker = a._tracker
        plain.append(np.asarray(a))
    result = np.concatenate(plain, axis=axis, **kwargs)
    if tracker is not None:
        name = _auto_name("concatenate")
        out = result.view(TrackedArray)
        out._tracker = tracker
        out._buf_name = name
        tracker.write(name, result.size)
        return out
    return result

"""Tests for auto-instrumented numpy wrapper (TrackedArray)."""

import math
import numpy as np
import pytest
from sparse_parity.tracker import MemTracker
from sparse_parity.lru_tracker import LRUStackTracker
from sparse_parity.tracked_numpy import TrackedArray, tracking_context, reset_counter


@pytest.fixture(autouse=True)
def _reset():
    reset_counter()
    yield
    reset_counter()


def make(arr, name="test", tracker=None):
    """Helper to create a TrackedArray."""
    if tracker is None:
        tracker = MemTracker()
    return TrackedArray(arr, name, tracker), tracker


# --- Basic creation and propagation ---

def test_creation_records_write():
    tracker = MemTracker()
    TrackedArray(np.zeros(10), "buf", tracker)
    s = tracker.summary()
    assert s["writes"] == 1
    assert s["total_floats_accessed"] == 10


def test_propagation_through_ufunc():
    tracker = MemTracker()
    a = TrackedArray(np.array([1, 2, 3]), "a", tracker)
    b = TrackedArray(np.array([4, 5, 6]), "b", tracker)
    c = a + b
    assert isinstance(c, TrackedArray)
    assert c._tracker is tracker


def test_propagation_through_xor():
    tracker = MemTracker()
    a = TrackedArray(np.array([1, 0, 1], dtype=np.uint8), "a", tracker)
    b = TrackedArray(np.array([0, 1, 1], dtype=np.uint8), "b", tracker)
    c = a ^ b
    assert isinstance(c, TrackedArray)
    np.testing.assert_array_equal(np.asarray(c), [1, 1, 0])


def test_propagation_through_comparison():
    tracker = MemTracker()
    a = TrackedArray(np.array([1, 0, 1], dtype=np.uint8), "a", tracker)
    result = a == 1
    assert isinstance(result, TrackedArray)


# --- Indexing ---

def test_getitem_tracks_slice_size():
    tracker = MemTracker()
    a = TrackedArray(np.arange(100), "a", tracker)
    row = a[10:20]
    assert isinstance(row, TrackedArray)
    # Read should be size 10 (the slice), not 100 (the whole array)
    reads = [(n, s, d) for t, n, s, _, d in tracker._events if t == "R"]
    assert len(reads) == 1
    assert reads[0][1] == 10  # size of slice read


def test_getitem_scalar_tracks_size_1():
    tracker = MemTracker()
    a = TrackedArray(np.arange(100), "a", tracker)
    val = a[5]
    # Scalar access should record a read of size 1
    reads = [(n, s, d) for t, n, s, _, d in tracker._events if t == "R"]
    assert len(reads) == 1
    assert reads[0][1] == 1


def test_setitem_tracks_write():
    tracker = MemTracker()
    a = TrackedArray(np.arange(10, dtype=np.uint8), "a", tracker)
    b = TrackedArray(np.array([99, 98, 97], dtype=np.uint8), "b", tracker)
    a[0:3] = b
    # Should record: read of b (size 3), write of a (size 3)
    s = tracker.summary()
    assert s["reads"] >= 1
    assert s["writes"] >= 3  # initial a, initial b, setitem write


def test_row_swap():
    tracker = MemTracker()
    arr = TrackedArray(
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint8), "arr", tracker
    )
    arr[[0, 1]] = arr[[1, 0]]
    np.testing.assert_array_equal(np.asarray(arr), [[3, 4], [1, 2], [5, 6]])


# --- tracking_context ---

def test_tracking_context_patches_zeros():
    tracker = MemTracker()
    with tracking_context(tracker):
        z = np.zeros((3, 4))
        assert isinstance(z, TrackedArray)
        assert z._tracker is tracker


def test_tracking_context_restores_zeros():
    tracker = MemTracker()
    with tracking_context(tracker):
        pass
    z = np.zeros((3, 4))
    assert not isinstance(z, TrackedArray)


def test_tracking_context_patches_ones():
    tracker = MemTracker()
    with tracking_context(tracker):
        o = np.ones(5)
        assert isinstance(o, TrackedArray)


# --- Copy and astype ---

def test_copy_preserves_tracking():
    tracker = MemTracker()
    a = TrackedArray(np.array([1, 2, 3]), "a", tracker)
    c = a.copy()
    assert isinstance(c, TrackedArray)
    assert c._tracker is tracker
    assert c._buf_name != a._buf_name  # different buffer


def test_astype_preserves_tracking():
    tracker = MemTracker()
    a = TrackedArray(np.array([1.0, 2.0]), "a", tracker)
    b = a.astype(np.uint8)
    assert isinstance(b, TrackedArray)
    assert b._tracker is tracker


# --- numpy functions ---

def test_np_where():
    tracker = MemTracker()
    a = TrackedArray(np.array([1, 0, 1, 0], dtype=np.uint8), "a", tracker)
    result = np.where(a == 1)
    # result is a tuple of arrays from np.where
    assert isinstance(result, tuple)


def test_np_prod():
    tracker = MemTracker()
    a = TrackedArray(np.array([[1, 2], [3, 4]]), "a", tracker)
    result = np.prod(a, axis=1)
    assert isinstance(result, TrackedArray)
    np.testing.assert_array_equal(np.asarray(result), [2, 12])


def test_np_sum():
    tracker = MemTracker()
    a = TrackedArray(np.array([1, 2, 3, 4]), "a", tracker)
    result = np.sum(a)
    assert result == 10


def test_np_all():
    tracker = MemTracker()
    a = TrackedArray(np.array([True, True, True]), "a", tracker)
    assert np.all(a) is np.bool_(True)


# --- Exact DMC prediction ---

def test_a_plus_b_plus_a_dmc():
    """Compute (a+b)+a with size-1 arrays and verify DMC matches prediction.

    Trace (each array is 1 float):

        clock=0: write("a", 1)     write_time["a"]=0   clock->1
        clock=1: write("b", 1)     write_time["b"]=1   clock->2

        c = a + b:
        clock=2: read("a", 1)      dist = 2-0 = 2      clock->3
        clock=3: read("b", 1)      dist = 3-1 = 2      clock->4
        clock=4: write(c, 1)       write_time[c]=4     clock->5

        d = c + a:
        clock=5: read(c, 1)        dist = 5-4 = 1      clock->6
        clock=6: read("a", 1)      dist = 6-0 = 6      clock->7
        clock=7: write(d, 1)       write_time[d]=7     clock->8

    DMC = sqrt(2) + sqrt(2) + sqrt(1) + sqrt(6) = 6.2779...
    ARD = (2 + 2 + 1 + 6) / 4 = 2.75

    For size-1 arrays, granular DMD equals DMC (each buffer is 1 float).
    """
    import math

    tracker = MemTracker()
    a = TrackedArray(np.array([1.0]), "a", tracker)
    b = TrackedArray(np.array([5.0]), "b", tracker)

    d = (a + b) + a

    np.testing.assert_array_equal(np.asarray(d), [7.0])

    s = tracker.summary()

    expected_dmc = math.sqrt(2) + math.sqrt(2) + math.sqrt(1) + math.sqrt(6)
    expected_ard = 2.75
    expected_reads = 4
    expected_writes = 4  # a, b, (a+b), (a+b)+a
    expected_total_floats = 8

    assert s["reads"] == expected_reads
    assert s["writes"] == expected_writes
    assert s["total_floats_accessed"] == expected_total_floats
    assert abs(s["weighted_ard"] - expected_ard) < 0.01
    assert abs(s["dmc"] - expected_dmc) < 0.01, \
        f"DMC {s['dmc']:.6f} != expected {expected_dmc:.6f}"


# --- GF(2) integration test ---

def test_gf2_gauss_elim_tracked():
    """Run the actual GF(2) algorithm with auto-tracking and verify correctness."""
    from sparse_parity.experiments.exp_gf2 import gf2_gauss_elim

    rng = np.random.RandomState(42)
    n_bits, k_sparse, n_samples = 20, 3, 21
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    A_raw = ((x + 1) / 2).astype(np.uint8)
    b_raw = ((y + 1) / 2).astype(np.uint8)

    tracker = MemTracker()
    with tracking_context(tracker):
        A = TrackedArray(A_raw, "A_gf2", tracker)
        b = TrackedArray(b_raw, "b_gf2", tracker)
        solution, rank = gf2_gauss_elim(A.copy(), b.copy())

    # Verify correctness
    predicted = sorted(np.where(np.asarray(solution) == 1)[0].tolist())
    assert predicted == secret

    # Verify tracking happened (should be many reads from pivot operations)
    s = tracker.summary()
    assert s["reads"] > 100  # many row operations
    assert s["writes"] > 100
    assert s["dmc"] > 0


def test_gf2_dmc_in_expected_range():
    """DMC should be in the ballpark of Yad's honest estimate (~189K)."""
    from sparse_parity.experiments.exp_gf2 import gf2_gauss_elim

    rng = np.random.RandomState(42)
    n_bits, k_sparse, n_samples = 20, 3, 21
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    A_raw = ((x + 1) / 2).astype(np.uint8)
    b_raw = ((y + 1) / 2).astype(np.uint8)

    tracker = MemTracker()
    with tracking_context(tracker):
        A = TrackedArray(A_raw, "A_gf2", tracker)
        b = TrackedArray(b_raw, "b_gf2", tracker)
        solution, rank = gf2_gauss_elim(A.copy(), b.copy())

    dmc = tracker.summary()["dmc"]
    # Should be order-of-magnitude consistent with honest estimate (~189K)
    # Auto-tracking gives ~227K due to intermediate buffer overhead
    assert 50_000 < dmc < 1_000_000, f"DMC {dmc} outside expected range"


# --- LRU stack tracker tests ---

def test_lru_paper_example():
    """Verify the paper's example: in 'abbbca', reuse distance of second a is 3."""
    t = LRUStackTracker()
    t.write('a', 1)   # a
    t.write('b', 1)   # b
    t.read('b', 1)    # b (at top, dist=1)
    t.read('b', 1)    # b (at top, dist=1)
    t.write('c', 1)   # c
    dists = t.read('a', 1)  # a (at position 3)
    assert dists[0] == 3


def test_lru_a_plus_b_plus_a():
    """(a+b)+a with LRU tracker. Reads observe stack without modifying it.

    Trace (size-1 arrays):

        write a: stack=[a]                        cold, dist=1
        write b: stack=[b,a]                      cold, dist=2

        c = a + b:
        read a:  a at pos 2, dist=2               stack unchanged [b,a]
        read b:  b at pos 1, dist=1               stack unchanged [b,a]
        write c: stack=[c,b,a]                    cold, dist=3

        d = c + a:
        read c:  c at pos 1, dist=1               stack unchanged [c,b,a]
        read a:  a at pos 3, dist=3               stack unchanged [c,b,a]
        write d: stack=[d,c,b,a]                  cold, dist=4

    Read DMD = sqrt(2) + sqrt(1) + sqrt(1) + sqrt(3) = 5.1463
    """
    tracker = LRUStackTracker()
    a = TrackedArray(np.array([1.0]), 'a', tracker)
    b = TrackedArray(np.array([5.0]), 'b', tracker)
    d = (a + b) + a

    np.testing.assert_array_equal(np.asarray(d), [7.0])

    s = tracker.summary()
    expected_read_dmd = math.sqrt(2) + math.sqrt(1) + math.sqrt(1) + math.sqrt(3)

    assert s['reads'] == 4
    assert s['writes'] == 4
    assert abs(s['read_dmd'] - expected_read_dmd) < 0.01, \
        f"read_dmd {s['read_dmd']:.6f} != expected {expected_read_dmd:.6f}"


def test_lru_gf2_integration():
    """GF(2) with LRU tracker gives read DMD close to honest estimate (~189K)."""
    from sparse_parity.experiments.exp_gf2 import gf2_gauss_elim

    rng = np.random.RandomState(42)
    n_bits, k_sparse, n_samples = 20, 3, 21
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    A_raw = ((x + 1) / 2).astype(np.uint8)
    b_raw = ((y + 1) / 2).astype(np.uint8)

    tracker = LRUStackTracker()
    with tracking_context(tracker):
        A = TrackedArray(A_raw, 'A_gf2', tracker)
        b = TrackedArray(b_raw, 'b_gf2', tracker)
        solution, rank = gf2_gauss_elim(A.copy(), b.copy())

    predicted = sorted(np.where(np.asarray(solution) == 1)[0].tolist())
    assert predicted == secret

    s = tracker.summary()
    read_dmd = s['read_dmd']
    assert 50_000 < read_dmd < 500_000, f"Read DMD {read_dmd} outside expected range"


# --- Additional coverage ---

def test_lru_cold_miss():
    """Reading an element that was never written is a cold miss."""
    t = LRUStackTracker()
    t.write('a', 1)  # stack = [a]
    dists = t.read('b', 1)  # b never written, cold miss
    assert dists[0] == 2  # stack has 1 element, so cold = len(stack) + 1 = 2
    assert t.summary()['cold_misses'] > 0


def test_transpose_is_zero_cost_view():
    """Transpose should not record a read or write (it's a view)."""
    tracker = MemTracker()
    a = TrackedArray(np.array([[1, 2], [3, 4]]), "a", tracker)
    writes_before = tracker.summary()['writes']
    b = a.T
    writes_after = tracker.summary()['writes']
    assert isinstance(b, TrackedArray)
    assert b._buf_name == a._buf_name  # same buffer
    assert writes_after == writes_before  # no new write


def test_tolist():
    tracker = MemTracker()
    a = TrackedArray(np.array([1, 2, 3]), "a", tracker)
    result = a.tolist()
    assert result == [1, 2, 3]
    assert tracker.summary()['reads'] >= 1


def test_np_mean():
    tracker = MemTracker()
    a = TrackedArray(np.array([1.0, 2.0, 3.0, 4.0]), "a", tracker)
    result = np.mean(a)
    assert result == 2.5
    assert tracker.summary()['reads'] >= 1


def test_np_sort():
    tracker = MemTracker()
    a = TrackedArray(np.array([3, 1, 2]), "a", tracker)
    result = np.sort(a)
    assert isinstance(result, TrackedArray)
    np.testing.assert_array_equal(np.asarray(result), [1, 2, 3])


def test_np_concatenate():
    tracker = MemTracker()
    a = TrackedArray(np.array([1, 2]), "a", tracker)
    b = TrackedArray(np.array([3, 4]), "b", tracker)
    result = np.concatenate([a, b])
    assert isinstance(result, TrackedArray)
    np.testing.assert_array_equal(np.asarray(result), [1, 2, 3, 4])


def test_np_zeros_like():
    tracker = MemTracker()
    a = TrackedArray(np.array([1, 2, 3]), "a", tracker)
    result = np.zeros_like(a)
    assert isinstance(result, TrackedArray)
    np.testing.assert_array_equal(np.asarray(result), [0, 0, 0])

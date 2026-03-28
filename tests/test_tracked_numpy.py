"""Tests for auto-instrumented numpy wrapper (TrackedArray)."""

import numpy as np
import pytest
from sparse_parity.tracker import MemTracker
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

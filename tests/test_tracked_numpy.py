"""Tests for auto-instrumented numpy wrapper (TrackedArray) and LRU tracker."""

import math
import numpy as np
import pytest
from sparse_parity.lru_tracker import LRUStackTracker
from sparse_parity.tracked_numpy import TrackedArray, tracking_context, reset_counter


@pytest.fixture(autouse=True)
def _reset():
    reset_counter()
    yield
    reset_counter()


@pytest.fixture
def gf2_data():
    """Generate GF(2) test data: n=20, k=3, 21 samples."""
    from sparse_parity.experiments.exp_gf2 import gf2_gauss_elim
    rng = np.random.RandomState(42)
    n_bits, k_sparse, n_samples = 20, 3, 21
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    A = ((x + 1) / 2).astype(np.uint8)
    b = ((y + 1) / 2).astype(np.uint8)
    return A, b, secret, gf2_gauss_elim


# =============================================================================
# TrackedArray: wrapper mechanics
# =============================================================================

class TestTrackedArrayPropagation:
    """TrackedArray wraps results so tracking propagates through operations."""

    def test_creation_records_write(self):
        tracker = LRUStackTracker()
        TrackedArray(np.zeros(10), "buf", tracker)
        assert tracker.summary()["writes"] == 1

    def test_ufunc(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([1, 2, 3]), "a", tracker)
        b = TrackedArray(np.array([4, 5, 6]), "b", tracker)
        c = a + b
        assert isinstance(c, TrackedArray)
        assert c._tracker is tracker

    def test_xor(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([1, 0, 1], dtype=np.uint8), "a", tracker)
        b = TrackedArray(np.array([0, 1, 1], dtype=np.uint8), "b", tracker)
        c = a ^ b
        assert isinstance(c, TrackedArray)
        np.testing.assert_array_equal(np.asarray(c), [1, 1, 0])

    def test_comparison(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([1, 0, 1], dtype=np.uint8), "a", tracker)
        assert isinstance(a == 1, TrackedArray)

    def test_copy(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([1, 2, 3]), "a", tracker)
        c = a.copy()
        assert isinstance(c, TrackedArray)
        assert c._tracker is tracker
        assert c._buf_name != a._buf_name

    def test_astype(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([1.0, 2.0]), "a", tracker)
        b = a.astype(np.uint8)
        assert isinstance(b, TrackedArray)
        assert b._tracker is tracker

    def test_transpose_is_zero_cost_view(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([[1, 2], [3, 4]]), "a", tracker)
        writes_before = tracker.summary()["writes"]
        b = a.T
        assert isinstance(b, TrackedArray)
        assert b._buf_name == a._buf_name  # same buffer, just a view
        assert tracker.summary()["writes"] == writes_before

    def test_tolist(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([1, 2, 3]), "a", tracker)
        assert a.tolist() == [1, 2, 3]
        assert tracker.summary()["reads"] >= 1


class TestTrackedArrayIndexing:
    """Indexing tracks the size of the actual slice accessed."""

    def test_slice_tracks_slice_size(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.arange(100), "a", tracker)
        row = a[10:20]
        assert isinstance(row, TrackedArray)
        reads = [(s, dists) for t, _, s, dists in tracker._events if t == "R"]
        assert reads[0][0] == 10  # size of slice, not 100

    def test_scalar_tracks_size_1(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.arange(100), "a", tracker)
        _ = a[5]
        reads = [(s, dists) for t, _, s, dists in tracker._events if t == "R"]
        assert reads[0][0] == 1

    def test_setitem(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.arange(10, dtype=np.uint8), "a", tracker)
        b = TrackedArray(np.array([99, 98, 97], dtype=np.uint8), "b", tracker)
        a[0:3] = b
        s = tracker.summary()
        assert s["reads"] >= 1
        assert s["writes"] >= 3  # initial a, initial b, setitem write

    def test_row_swap(self):
        tracker = LRUStackTracker()
        arr = TrackedArray(
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint8), "arr", tracker
        )
        arr[[0, 1]] = arr[[1, 0]]
        np.testing.assert_array_equal(np.asarray(arr), [[3, 4], [1, 2], [5, 6]])


class TestTrackingContext:
    """tracking_context patches numpy constructors to return TrackedArrays."""

    def test_patches_zeros(self):
        tracker = LRUStackTracker()
        with tracking_context(tracker):
            z = np.zeros((3, 4))
            assert isinstance(z, TrackedArray)
            assert z._tracker is tracker

    def test_restores_zeros(self):
        tracker = LRUStackTracker()
        with tracking_context(tracker):
            pass
        assert not isinstance(np.zeros((3, 4)), TrackedArray)

    def test_patches_ones(self):
        tracker = LRUStackTracker()
        with tracking_context(tracker):
            assert isinstance(np.ones(5), TrackedArray)


class TestNumpyFunctions:
    """numpy functions on TrackedArrays record reads and wrap outputs."""

    def test_where(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([1, 0, 1, 0], dtype=np.uint8), "a", tracker)
        result = np.where(a == 1)
        assert isinstance(result, tuple)

    def test_prod(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([[1, 2], [3, 4]]), "a", tracker)
        result = np.prod(a, axis=1)
        assert isinstance(result, TrackedArray)
        np.testing.assert_array_equal(np.asarray(result), [2, 12])

    def test_sum(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([1, 2, 3, 4]), "a", tracker)
        assert np.sum(a) == 10

    def test_all(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([True, True, True]), "a", tracker)
        assert np.all(a) is np.bool_(True)

    def test_mean(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([1.0, 2.0, 3.0, 4.0]), "a", tracker)
        assert np.mean(a) == 2.5
        assert tracker.summary()["reads"] >= 1

    def test_sort(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([3, 1, 2]), "a", tracker)
        result = np.sort(a)
        assert isinstance(result, TrackedArray)
        np.testing.assert_array_equal(np.asarray(result), [1, 2, 3])

    def test_concatenate(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([1, 2]), "a", tracker)
        b = TrackedArray(np.array([3, 4]), "b", tracker)
        result = np.concatenate([a, b])
        assert isinstance(result, TrackedArray)
        np.testing.assert_array_equal(np.asarray(result), [1, 2, 3, 4])

    def test_zeros_like(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([1, 2, 3]), "a", tracker)
        result = np.zeros_like(a)
        assert isinstance(result, TrackedArray)
        np.testing.assert_array_equal(np.asarray(result), [0, 0, 0])


# =============================================================================
# LRU Stack Tracker: metric correctness (Ding et al., Definition 2.1)
# =============================================================================

class TestLRUPaperExample:
    """Verify against the paper: in 'abbbca', reuse distance of second a is 3."""

    def test_abbbca(self):
        t = LRUStackTracker()
        t.write('a', 1)
        t.write('b', 1)
        t.read('b', 1)
        t.read('b', 1)
        t.write('c', 1)
        dists = t.read('a', 1)
        assert dists[0] == 3

    def test_unwritten_read(self):
        """Reading an element never written: dist = len(stack) + 1."""
        t = LRUStackTracker()
        t.write('a', 1)  # stack = [a], size 1
        dists = t.read('b', 1)  # b not in stack
        assert dists[0] == 2  # len(stack) + 1

    def test_writes_are_free(self):
        """Writes place data on stack but do not accumulate DMD."""
        t = LRUStackTracker()
        t.write('a', 10)
        t.write('b', 10)
        t.write('c', 10)
        assert t.summary()['dmd'] == 0.0  # no reads, no cost


class TestLRUMetricPrediction:
    """Exact DMD predictions for simple expressions.

    (a+b)+a with size-1 arrays. Each array is a single float.
    Only reads cost DMD. Writes just place data on the stack.

        write a: stack=[a]                        free
        write b: stack=[a,b]                      free

        c = a + b:
        read a:  a at pos 2, dist=2               stack unchanged [a,b]
        read b:  b at pos 1, dist=1               stack unchanged [a,b]
        write c: stack=[a,b,c]                    free

        d = c + a:
        read c:  c at pos 1, dist=1               stack unchanged [a,b,c]
        read a:  a at pos 3, dist=3               stack unchanged [a,b,c]
        write d: stack=[a,b,c,d]                  free

    DMD = sqrt(2) + sqrt(1) + sqrt(1) + sqrt(3) = 5.1463
    """

    def test_a_plus_b_plus_a(self):
        tracker = LRUStackTracker()
        a = TrackedArray(np.array([1.0]), 'a', tracker)
        b = TrackedArray(np.array([5.0]), 'b', tracker)
        d = (a + b) + a

        np.testing.assert_array_equal(np.asarray(d), [7.0])

        s = tracker.summary()
        expected = math.sqrt(2) + math.sqrt(1) + math.sqrt(1) + math.sqrt(3)

        assert s['reads'] == 4
        assert s['writes'] == 4
        assert abs(s['dmd'] - expected) < 0.01, \
            f"dmd {s['dmd']:.4f} != expected {expected:.4f}"


class TestGF2Integration:
    """GF(2) Gaussian elimination with auto-tracking."""

    def test_correctness(self, gf2_data):
        """Algorithm still finds the correct secret with tracking enabled."""
        A, b, secret, gf2_gauss_elim = gf2_data
        tracker = LRUStackTracker()
        with tracking_context(tracker):
            solution, rank = gf2_gauss_elim(
                TrackedArray(A, 'A', tracker).copy(),
                TrackedArray(b, 'b', tracker).copy(),
            )
        predicted = sorted(np.where(np.asarray(solution) == 1)[0].tolist())
        assert predicted == secret

    def test_dmd_in_expected_range(self, gf2_data):
        """DMD should be near Yad's honest estimate (~189K)."""
        A, b, secret, gf2_gauss_elim = gf2_data
        tracker = LRUStackTracker()
        with tracking_context(tracker):
            gf2_gauss_elim(
                TrackedArray(A, 'A', tracker).copy(),
                TrackedArray(b, 'b', tracker).copy(),
            )
        dmd = tracker.summary()['dmd']
        assert 50_000 < dmd < 500_000, f"DMD {dmd} outside expected range"

    def test_many_operations_tracked(self, gf2_data):
        """Elimination should generate many reads/writes from pivot operations."""
        A, b, _, gf2_gauss_elim = gf2_data
        tracker = LRUStackTracker()
        with tracking_context(tracker):
            gf2_gauss_elim(
                TrackedArray(A, 'A', tracker).copy(),
                TrackedArray(b, 'b', tracker).copy(),
            )
        s = tracker.summary()
        assert s['reads'] > 100
        assert s['writes'] > 100

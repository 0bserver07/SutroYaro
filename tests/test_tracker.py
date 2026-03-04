from sparse_parity.tracker import MemTracker


def test_write_read_distance():
    t = MemTracker()
    t.write('a', 100)    # clock: 0 -> 100
    t.write('b', 50)     # clock: 100 -> 150
    dist = t.read('a')   # clock: 150, distance = 150 - 0 = 150
    assert dist == 150


def test_clock_advances_by_size():
    t = MemTracker()
    t.write('a', 100)
    t.write('b', 200)
    assert t.clock == 300


def test_read_unknown_returns_negative():
    t = MemTracker()
    dist = t.read('nonexistent', 10)
    assert dist == -1


def test_weighted_ard():
    t = MemTracker()
    t.write('small', 1)    # clock 0->1
    t.write('big', 1000)   # clock 1->1001
    t.read('small', 1)     # dist=1001, 1 float
    t.read('big', 1000)    # dist=1000, 1000 floats
    summary = t.summary()
    # Weighted avg dominated by 'big' (1000 floats at dist 1000)
    # vs 'small' (1 float at dist 1001)
    # = (1*1001 + 1000*1000) / (1 + 1000) = 1001001/1001 ≈ 1000
    assert 999 < summary['weighted_ard'] < 1002


def test_to_json_has_required_fields():
    t = MemTracker()
    t.write('x', 10)
    t.read('x')
    j = t.to_json()
    assert 'total_floats_accessed' in j
    assert 'reads' in j
    assert 'writes' in j
    assert 'weighted_ard' in j
    assert 'per_buffer' in j

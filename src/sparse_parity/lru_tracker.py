"""True LRU stack distance tracker for granular DMD (Ding et al., arXiv:2312.14441).

Unlike MemTracker which uses a clock-based approximation, this tracks every
individual float in an LRU stack and computes per-element stack distances.

From the paper: "in abbbca, the reuse distance of the second a is 3. Its DMD
is sqrt(3)." The stack distance is the 1-indexed position of the element in
the LRU stack at the time of access.

DMD = sqrt(stack_distance) for each access.
DMC = sum of all DMDs.
"""

import math


class LRUStackTracker:
    """Tracks per-element LRU stack distances and computes granular DMD.

    Each float is identified by (buffer_name, index). The LRU stack maintains
    all written elements ordered by recency of writes. Writes move the element
    to the top of the stack. Reads observe the element's 1-indexed position
    without moving it.

    API matches MemTracker: write(name, size), read(name, size).
    """

    def __init__(self):
        # LRU stack: index 0 = most recently accessed
        self._stack = []
        # Fast lookup: element_id -> index in _stack
        self._pos = {}
        self._total_dmd = 0.0
        self._total_cold_dmd = 0.0
        self._events = []  # (type, name, size, distances_list)
        self._n_reads = 0
        self._n_writes = 0
        self._n_accesses = 0  # individual element accesses
        self._n_cold = 0

    def _write_element(self, element_id):
        """Write one element. Moves it to top of LRU stack.

        Returns (stack_distance, is_cold).
        """
        if element_id in self._pos:
            idx = self._pos[element_id]
            dist = idx + 1  # 1-indexed
            self._stack.pop(idx)
            for i in range(idx, len(self._stack)):
                self._pos[self._stack[i]] = i
            is_cold = False
        else:
            dist = len(self._stack) + 1
            is_cold = True

        self._stack.insert(0, element_id)
        for i in range(len(self._stack)):
            self._pos[self._stack[i]] = i

        return dist, is_cold

    def _read_element(self, element_id):
        """Read one element. Observes its stack position without moving it.

        Returns (stack_distance, is_cold).
        """
        if element_id in self._pos:
            idx = self._pos[element_id]
            dist = idx + 1  # 1-indexed
            return dist, False
        else:
            dist = len(self._stack) + 1
            return dist, True

    def write(self, name, size):
        """Write size floats to buffer. Each float is pushed onto the LRU stack."""
        distances = []
        for i in range(size):
            dist, is_cold = self._write_element((name, i))
            dmd = math.sqrt(dist)
            self._total_dmd += dmd
            if is_cold:
                self._total_cold_dmd += dmd
                self._n_cold += 1
            distances.append(dist)
            self._n_accesses += 1
        self._events.append(('W', name, size, distances))
        self._n_writes += 1

    def read(self, name, size=None):
        """Read size floats from buffer. Observes stack positions without moving.

        Returns list of per-element stack distances.
        """
        if size is None:
            # Infer from last write
            for typ, n, s, _ in reversed(self._events):
                if typ == 'W' and n == name:
                    size = s
                    break
            else:
                size = 0
        distances = []
        for i in range(size):
            dist, is_cold = self._read_element((name, i))
            dmd = math.sqrt(dist)
            self._total_dmd += dmd
            if is_cold:
                self._total_cold_dmd += dmd
                self._n_cold += 1
            distances.append(dist)
            self._n_accesses += 1
        self._events.append(('R', name, size, distances))
        self._n_reads += 1
        return distances

    def summary(self):
        """Return summary metrics."""
        read_events = [(n, s, dists) for typ, n, s, dists in self._events if typ == 'R']

        per_buffer = {}
        for name, size, dists in read_events:
            if name not in per_buffer:
                per_buffer[name] = {'size': size, 'distances': []}
            per_buffer[name]['distances'].extend(dists)

        for name, info in per_buffer.items():
            dists = info['distances']
            if dists:
                info['avg_dist'] = sum(dists) / len(dists)
                info['min_dist'] = min(dists)
                info['max_dist'] = max(dists)
                info['read_count'] = len(dists)
                info['dmd'] = sum(math.sqrt(d) for d in dists)

        total_read_dmd = sum(info.get('dmd', 0) for info in per_buffer.values())

        return {
            'granular_dmd': self._total_dmd,
            'read_dmd': total_read_dmd,
            'cold_dmd': self._total_cold_dmd,
            'total_accesses': self._n_accesses,
            'reads': self._n_reads,
            'writes': self._n_writes,
            'cold_misses': self._n_cold,
            'stack_size': len(self._stack),
            'per_buffer': per_buffer,
        }

    def to_json(self):
        return self.summary()

    def report(self):
        s = self.summary()
        print(f"\n{'=' * 70}")
        print(f"  LRU STACK DISTANCE REPORT (Granular DMD)")
        print(f"{'=' * 70}")
        print(f"  Total DMD (all accesses): {s['granular_dmd']:,.1f}")
        print(f"  Read DMD: {s['read_dmd']:,.1f}")
        print(f"  Cold miss DMD: {s['cold_dmd']:,.1f}")
        print(f"  Element accesses: {s['total_accesses']:,}")
        print(f"  Operations: {s['reads']} reads, {s['writes']} writes")
        print(f"  Cold misses: {s['cold_misses']:,}")
        print(f"  Stack size: {s['stack_size']:,}")
        if s['per_buffer']:
            print(f"\n  {'Buffer':<20} {'Elems':>6} {'Reads':>6} {'Avg Dist':>10} {'DMD':>10}")
            print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*10} {'─'*10}")
            for name, info in sorted(s['per_buffer'].items(),
                                     key=lambda x: -x[1].get('dmd', 0)):
                if 'read_count' in info:
                    print(f"  {name:<20} {info['size']:>6} {info['read_count']:>6} "
                          f"{info['avg_dist']:>10,.1f} {info['dmd']:>10,.1f}")
        print(f"{'=' * 70}")

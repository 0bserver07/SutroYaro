"""True LRU stack distance tracker for granular DMD (Ding et al., arXiv:2312.14441).

Tracks every individual float in an LRU stack and computes per-element
stack distances. Only reads incur DMD cost. Writes place data on the
stack (moving elements to the top) but are free -- there are no "cold
misses" because inputs are assumed to arrive pre-loaded on the stack.

DMD = sqrt(stack_distance) for each read.
DMC = sum of all read DMDs.
"""

import math
from collections import defaultdict


class LRUStackTracker:
    """Tracks per-element LRU stack distances and computes granular DMD.

    Each float is identified by (buffer_name, index). Writes place elements
    at the top of the stack (free -- no DMD cost). Reads observe each
    element's 1-indexed stack position and accumulate DMD = sqrt(distance).

    API matches MemTracker: write(name, size), read(name, size).
    """

    def __init__(self):
        self._stack = []        # index 0 = most recently written
        self._pos = {}          # element_id -> index in _stack
        self._dmd = 0.0         # total DMD (reads only)
        self._events = []       # (type, name, size, distances_list)
        self._n_reads = 0
        self._n_writes = 0

    def _write_element(self, element_id):
        """Write one element. Moves it to top of LRU stack. Returns stack_distance."""
        if element_id in self._pos:
            idx = self._pos[element_id]
            dist = idx + 1
            self._stack.pop(idx)
            self._stack.insert(0, element_id)
            for i in range(idx + 1):
                self._pos[self._stack[i]] = i
        else:
            dist = len(self._stack) + 1
            self._stack.insert(0, element_id)
            for i in range(len(self._stack)):
                self._pos[self._stack[i]] = i
        return dist

    def _read_element(self, element_id):
        """Read one element. Observes stack position without moving. Returns stack_distance."""
        if element_id in self._pos:
            return self._pos[element_id] + 1
        return len(self._stack) + 1

    def write(self, name, size):
        """Write size floats to buffer. Each float is pushed onto the LRU stack.

        Writes are free (no DMD cost). They only update the stack state.
        """
        distances = [self._write_element((name, i)) for i in range(size)]
        self._events.append(('W', name, size, distances))
        self._n_writes += 1

    def read(self, name, size=None):
        """Read size floats from buffer. Each read accumulates DMD = sqrt(distance).

        Returns list of per-element stack distances.
        """
        if size is None:
            for typ, n, s, _ in reversed(self._events):
                if typ == 'W' and n == name:
                    size = s
                    break
            else:
                size = 0
        distances = []
        for i in range(size):
            dist = self._read_element((name, i))
            self._dmd += math.sqrt(dist)
            distances.append(dist)
        self._events.append(('R', name, size, distances))
        self._n_reads += 1
        return distances

    def summary(self):
        """Return summary metrics."""
        per_buffer = defaultdict(lambda: {'distances': []})
        for typ, name, size, dists in self._events:
            if typ == 'R':
                per_buffer[name]['size'] = size
                per_buffer[name]['distances'].extend(dists)

        for info in per_buffer.values():
            dists = info['distances']
            if dists:
                info['avg_dist'] = sum(dists) / len(dists)
                info['min_dist'] = min(dists)
                info['max_dist'] = max(dists)
                info['read_count'] = len(dists)
                info['dmd'] = sum(math.sqrt(d) for d in dists)

        return {
            'dmd': self._dmd,
            'reads': self._n_reads,
            'writes': self._n_writes,
            'stack_size': len(self._stack),
            'per_buffer': dict(per_buffer),
        }

    def to_json(self):
        return self.summary()

    def report(self):
        s = self.summary()
        print(f"\n{'=' * 70}")
        print(f"  LRU STACK DISTANCE REPORT")
        print(f"{'=' * 70}")
        print(f"  DMD (reads only): {s['dmd']:,.1f}")
        print(f"  Operations: {s['reads']} reads, {s['writes']} writes")
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

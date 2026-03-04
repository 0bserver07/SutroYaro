"""Phase 3: Memory Reuse Distance Tracker for energy efficiency measurement."""


class MemTracker:
    """
    Tracks Average Reuse Distance (ARD) — a proxy for energy efficiency.

    Clock advances by buffer SIZE (floats), not operation count.
    Small ARD = data stays in cache = cheap.
    Large ARD = cache miss = expensive external memory access.
    """

    def __init__(self):
        self.clock = 0
        self._write_time = {}
        self._write_size = {}
        self._events = []

    def write(self, name, size):
        """Record writing `size` floats to buffer `name`."""
        self._write_time[name] = self.clock
        self._write_size[name] = size
        self._events.append(('W', name, size, self.clock, None))
        self.clock += size

    def read(self, name, size=None):
        """Record reading from buffer `name`. Returns reuse distance."""
        if size is None:
            size = self._write_size.get(name, 0)
        if name in self._write_time:
            distance = self.clock - self._write_time[name]
        else:
            distance = -1
        self._events.append(('R', name, size, self.clock, distance))
        self.clock += size
        return distance

    def summary(self):
        """Compute summary statistics."""
        reads = [(name, size, dist) for typ, name, size, _, dist in self._events
                 if typ == 'R' and dist >= 0]
        writes = [e for e in self._events if e[0] == 'W']

        if not reads:
            return {'total_floats_accessed': self.clock, 'reads': 0, 'writes': len(writes),
                    'weighted_ard': 0, 'per_buffer': {}}

        total_float_dist = sum(s * d for _, s, d in reads)
        total_floats = sum(s for _, s, _ in reads)
        weighted_ard = total_float_dist / total_floats if total_floats > 0 else 0

        per_buffer = {}
        for name, size, dist in reads:
            if name not in per_buffer:
                per_buffer[name] = {'size': size, 'distances': []}
            per_buffer[name]['distances'].append(dist)

        for name, info in per_buffer.items():
            dists = info['distances']
            info['avg_dist'] = sum(dists) / len(dists)
            info['min_dist'] = min(dists)
            info['max_dist'] = max(dists)
            info['read_count'] = len(dists)

        return {
            'total_floats_accessed': self.clock,
            'reads': len(reads),
            'writes': len(writes),
            'weighted_ard': weighted_ard,
            'total_floats_read': total_floats,
            'per_buffer': per_buffer,
        }

    def to_json(self):
        """Return JSON-serializable dict of all metrics."""
        return self.summary()

    def report(self):
        """Print human-readable report."""
        s = self.summary()
        print(f"\n{'=' * 70}")
        print(f"  MEMORY REUSE DISTANCE REPORT")
        print(f"{'=' * 70}")
        print(f"  Total floats accessed: {s['total_floats_accessed']:,}")
        print(f"  Operations: {s['reads']} reads, {s['writes']} writes")
        print(f"  Weighted ARD: {s['weighted_ard']:,.0f} floats")
        if s['per_buffer']:
            print(f"\n  {'Buffer':<12} {'Size':>8} {'Reads':>5} {'Avg Dist':>10} {'Min':>8} {'Max':>8}")
            print(f"  {'─'*12} {'─'*8} {'─'*5} {'─'*10} {'─'*8} {'─'*8}")
            for name, info in s['per_buffer'].items():
                print(f"  {name:<12} {info['size']:>8,} {info['read_count']:>5} "
                      f"{info['avg_dist']:>10,.0f} {info['min_dist']:>8,} {info['max_dist']:>8,}")
        print(f"{'=' * 70}")

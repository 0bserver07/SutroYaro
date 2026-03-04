"""Cache-aware memory tracker — extends MemTracker with LRU cache simulation."""

from collections import OrderedDict
from sparse_parity.tracker import MemTracker


class CacheTracker(MemTracker):
    """
    MemTracker with LRU cache simulation.

    Simulates a cache of `cache_size_floats` capacity. On each read:
    - If the buffer fits in cache AND was recently accessed (still resident): HIT (zero cost).
    - Otherwise: MISS (full reuse-distance cost, buffer loaded into cache).

    LRU eviction: oldest-accessed buffers are evicted first when cache is full.
    """

    def __init__(self, cache_size_floats):
        super().__init__()
        self.cache_size = cache_size_floats
        # OrderedDict: name -> size. Most-recently-used at end.
        self._cache = OrderedDict()
        self._cache_used = 0
        self.hits = 0
        self.misses = 0
        self._hit_events = []   # (name, size, clock)
        self._miss_events = []  # (name, size, clock, distance)

    def _evict_until(self, needed):
        """Evict LRU entries until `needed` floats can fit."""
        while self._cache_used + needed > self.cache_size and self._cache:
            evicted_name, evicted_size = self._cache.popitem(last=False)
            self._cache_used -= evicted_size

    def _cache_touch(self, name, size):
        """Move buffer to MRU position (or insert it)."""
        if name in self._cache:
            # Already in cache — move to end (MRU)
            self._cache.move_to_end(name)
            return True  # hit
        # Not in cache — need to insert
        if size > self.cache_size:
            # Buffer bigger than entire cache — always a miss, don't cache
            return False
        self._evict_until(size)
        self._cache[name] = size
        self._cache_used += size
        return False  # miss (just loaded)

    def read(self, name, size=None):
        """Record reading from buffer `name`. Returns reuse distance.

        Also tracks cache hit/miss.
        """
        if size is None:
            size = self._write_size.get(name, 0)

        # Check cache BEFORE advancing clock
        in_cache = name in self._cache

        if in_cache:
            # HIT: buffer is resident
            self.hits += 1
            self._hit_events.append((name, size, self.clock))
            # Touch to update LRU position
            self._cache.move_to_end(name)
            # Still record the event in parent for full ARD tracking
            distance = super().read(name, size)
            return distance
        else:
            # MISS: load into cache
            self.misses += 1
            distance = super().read(name, size)
            self._miss_events.append((name, size, self.clock, distance))
            # Insert into cache (may evict)
            self._cache_touch(name, size)
            return distance

    def write(self, name, size):
        """Record writing `size` floats to buffer `name`.

        Writes also update cache state — the written buffer becomes resident.
        """
        super().write(name, size)
        # Writing a buffer makes it resident in cache
        if name in self._cache:
            # Update size if changed, move to MRU
            old_size = self._cache[name]
            self._cache_used -= old_size
            self._cache.move_to_end(name)
            self._cache[name] = size
            self._cache_used += size
        else:
            if size <= self.cache_size:
                self._evict_until(size)
                self._cache[name] = size
                self._cache_used += size

    def cache_summary(self):
        """Return cache-specific metrics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        # Effective ARD: weighted ARD counting only misses
        miss_reads = [(name, size, dist) for name, size, _, dist in self._miss_events
                      if dist >= 0]
        if miss_reads:
            total_miss_float_dist = sum(s * d for _, s, d in miss_reads)
            total_miss_floats = sum(s for _, s, _ in miss_reads)
            effective_ard = total_miss_float_dist / total_miss_floats if total_miss_floats > 0 else 0
        else:
            effective_ard = 0
            total_miss_floats = 0

        return {
            'cache_size_floats': self.cache_size,
            'hits': self.hits,
            'misses': self.misses,
            'total_accesses': total,
            'hit_rate': hit_rate,
            'effective_ard': effective_ard,
            'total_miss_floats': total_miss_floats,
        }

    def to_json(self):
        """Return JSON-serializable dict of all metrics including cache stats."""
        result = super().to_json()
        result['cache'] = self.cache_summary()
        return result

    def report(self):
        """Print human-readable report with cache statistics."""
        super().report()
        c = self.cache_summary()
        print(f"\n  CACHE SIMULATION (capacity: {c['cache_size_floats']:,} floats = "
              f"{c['cache_size_floats'] * 4 / 1024:.0f} KB)")
        print(f"  Hits: {c['hits']:,}  Misses: {c['misses']:,}  "
              f"Hit rate: {c['hit_rate']:.1%}")
        print(f"  Effective ARD (misses only): {c['effective_ard']:,.0f} floats")
        print(f"{'=' * 70}")

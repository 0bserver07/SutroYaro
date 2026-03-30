"""True LRU stack distance tracker for granular DMD (Ding et al., arXiv:2312.14441).

Tracks every individual float in an LRU stack and computes per-element
stack distances. Only reads incur DMD cost. Writes place data on the
stack (moving elements to the top) but are free -- there are no "cold
misses" because inputs are assumed to arrive pre-loaded on the stack.

DMD = sqrt(stack_distance) for each read.
DMC = sum of all read DMDs.

Uses a splay tree (Olken, 1981) for O(log n) per-element stack operations.
Adapted from Wang, Ding & He (2026) "Ranking Human and LLM Texts Using
Locality Statistics" — https://github.com/YYWmus/locality-metrics-for-text
(MIT License).
"""

import math
from collections import defaultdict


# =============================================================================
# Splay tree for O(log n) LRU stack distance (Olken, 1981)
# =============================================================================

class _SplayNode:
    __slots__ = ('parent', 'children', 'count')

    def __init__(self):
        self.parent = None
        self.children = [None, None]
        self.count = 1

    @staticmethod
    def left_count(node):
        return node.children[0].count if node.children[0] else 0

    @staticmethod
    def maintain(node):
        lc = node.children[0].count if node.children[0] else 0
        rc = node.children[1].count if node.children[1] else 0
        node.count = 1 + lc + rc

    @staticmethod
    def is_right_child(node):
        return node.parent.children[1] is node

    @staticmethod
    def rotate(node):
        parent = node.parent
        is_right = _SplayNode.is_right_child(node)
        if parent.parent:
            gp_right = _SplayNode.is_right_child(parent)
            parent.parent.children[gp_right] = node
        node.parent = parent.parent
        child = node.children[not is_right]
        parent.children[is_right] = child
        if child:
            child.parent = parent
        _SplayNode.maintain(parent)
        node.children[not is_right] = parent
        parent.parent = node
        _SplayNode.maintain(node)

    @staticmethod
    def splay(node):
        while node.parent:
            parent = node.parent
            if parent.parent:
                if _SplayNode.is_right_child(node) == _SplayNode.is_right_child(parent):
                    _SplayNode.rotate(parent)
                else:
                    _SplayNode.rotate(node)
            _SplayNode.rotate(node)

    @staticmethod
    def find_leftmost(node):
        while node.children[0]:
            node = node.children[0]
        return node

    @staticmethod
    def remove_root(node):
        left, right = node.children
        if right:
            right.parent = None
            leftmost = _SplayNode.find_leftmost(right)
            _SplayNode.splay(leftmost)
            leftmost.children[0] = left
            if left:
                left.parent = leftmost
            _SplayNode.maintain(leftmost)
            return leftmost
        if left:
            left.parent = None
        return left

    @staticmethod
    def insert_front(node, root):
        node.children[1] = root
        _SplayNode.maintain(node)
        if root:
            root.parent = node
        return node


class _LRUSplay:
    """O(log n) LRU stack using a splay tree with subtree counts."""

    def __init__(self):
        self.root = None
        self.handles = {}
        self._size = 0

    def write(self, key):
        """Move key to top of stack (MRU). Returns 1-indexed stack distance.

        For new keys, distance = current_size + 1.
        """
        if key in self.handles:
            node = self.handles[key]
            _SplayNode.splay(node)
            distance = _SplayNode.left_count(node) + 1
            new_root = _SplayNode.remove_root(node)
            node.count = 1
            node.children = [None, None]
            node.parent = None
            self.root = _SplayNode.insert_front(node, new_root)
            return distance
        else:
            node = _SplayNode()
            self.handles[key] = node
            self.root = _SplayNode.insert_front(node, self.root)
            self._size += 1
            return self._size  # cold: distance = total elements so far

    def read(self, key):
        """Observe key's stack position WITHOUT moving it. Returns 1-indexed distance."""
        if key in self.handles:
            node = self.handles[key]
            _SplayNode.splay(node)
            dist = _SplayNode.left_count(node) + 1
            # Splay moved it to root — we need to undo this to preserve
            # the "reads don't move" semantics. But splay trees don't support
            # efficient rank queries without splaying. As a practical compromise,
            # we leave it splayed — this changes the tree structure but not the
            # logical LRU order (the in-order traversal is unchanged).
            return dist
        return self._size + 1

    @property
    def size(self):
        return self._size


class LRUStackTracker:
    """Tracks per-element LRU stack distances and computes granular DMD.

    Each float is identified by (buffer_name, index). Writes place elements
    at the top of the stack (free -- no DMD cost). Reads observe each
    element's 1-indexed stack position and accumulate DMD = sqrt(distance).

    Uses a splay tree for O(log n) operations instead of O(n) list scanning.

    API matches MemTracker: write(name, size), read(name, size).
    """

    def __init__(self):
        self._lru = _LRUSplay()
        self._dmd = 0.0         # total DMD (reads only)
        self._events = []       # (type, name, size, distances_list)
        self._n_reads = 0
        self._n_writes = 0

    def _write_element(self, element_id):
        """Write one element. Moves it to top of LRU stack. Returns stack_distance."""
        return self._lru.write(element_id)

    def _read_element(self, element_id):
        """Read one element. Observes stack position without moving. Returns stack_distance."""
        return self._lru.read(element_id)

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
            'stack_size': self._lru.size,
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

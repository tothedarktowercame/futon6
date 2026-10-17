"""Interval queries over (start, end) extents, shared by the detector and the checker.

Pure data structures, no detector logic. Both sides ask, once per mark or span,
whether any extent contains a range, overlaps it or holds a point, and which
extents strictly contain a point. Scanning every extent for every question is
quadratic in a long paper; these answer each question in logarithmic time and
give exactly the answers the scans gave.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from itertools import accumulate


class Extents:
    """any(...) over a fixed set of extents; their order and overlaps do not matter."""

    def __init__(self, extents):
        ordered = sorted((a, b) for a, b in extents)
        self._starts = [a for a, _b in ordered]
        # the furthest end among the extents starting at or before each position
        self._reach = list(accumulate((b for _a, b in ordered), max))

    def contain(self, s, e):
        """any(a <= s and e <= b for a, b in extents)"""
        count = bisect_right(self._starts, s)
        return count > 0 and self._reach[count - 1] >= e

    def overlap(self, s, e):
        """any(a < e and b > s for a, b in extents)"""
        count = bisect_left(self._starts, e)
        return count > 0 and self._reach[count - 1] > s

    def hold(self, p):
        """any(a <= p < b for a, b in extents)"""
        count = bisect_right(self._starts, p)
        return count > 0 and self._reach[count - 1] > p


class Stabbing:
    """The extents strictly containing a point, by their position in the list given.

    Extents sorted by start carry a max-end tree, so a query visits only the
    subtrees that can hold a containing extent.
    """

    def __init__(self, extents):
        self.extents = list(extents)
        self._order = sorted(range(len(self.extents)), key=lambda j: self.extents[j][0])
        self._starts = [self.extents[j][0] for j in self._order]
        self._size = 1
        while self._size < len(self._order):
            self._size *= 2
        self._max_end = [float("-inf")] * (2 * self._size)
        for leaf, j in enumerate(self._order):
            self._max_end[self._size + leaf] = self.extents[j][1]
        for node in range(self._size - 1, 0, -1):
            self._max_end[node] = max(self._max_end[2 * node], self._max_end[2 * node + 1])

    def containing(self, x):
        """The list positions j with extents[j][0] < x < extents[j][1]."""
        return self.meeting(x, x)

    def meeting(self, lo_, hi_):
        """The list positions j with extents[j][0] < hi_ and extents[j][1] > lo_."""
        below = bisect_left(self._starts, hi_)  # sorted extents [0, below) start before hi_
        found, stack = [], [(1, 0, self._size)]
        while stack:
            node, lo, hi = stack.pop()
            if lo >= below or self._max_end[node] <= lo_:
                continue
            if hi - lo == 1:
                found.append(self._order[lo])
            else:
                mid = (lo + hi) // 2
                stack.extend(((2 * node, lo, mid), (2 * node + 1, mid, hi)))
        return found

    def settle(self, x, to_end):
        """X after one pass over the extents in list order, where each extent
        strictly containing the current X moves it to that extent's end (to_end)
        or start."""
        last = -1
        while True:
            later = [j for j in self.containing(x) if j > last]
            if not later:
                return x
            last = min(later)
            x = self.extents[last][1] if to_end else self.extents[last][0]

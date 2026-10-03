"""The shared interval queries answer exactly what scanning every extent answered."""
from __future__ import annotations

from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dp_capabilities.intervals import Extents, Stabbing  # noqa: E402
from dp_capabilities import wellformed  # noqa: E402


def random_extents(rng: random.Random) -> list[tuple[int, int]]:
    # empty, reversed, nested and overlapping extents all occur in real marks
    return [(a, a + rng.randint(-3, 12)) for a in (rng.randint(-2, 40) for _ in range(rng.randint(0, 14)))]


class IntervalQueryTests(unittest.TestCase):
    def test_extents_match_the_scans_they_replace(self):
        rng = random.Random(1)
        for _ in range(3000):
            extents = random_extents(rng)
            ex = Extents(extents)
            for _ in range(10):
                s = rng.randint(-4, 46)
                e = s + rng.randint(-4, 12)
                self.assertEqual(ex.contain(s, e), any(a <= s and e <= b for a, b in extents))
                self.assertEqual(ex.overlap(s, e), any(a < e and b > s for a, b in extents))
                self.assertEqual(ex.hold(s), any(a <= s < b for a, b in extents))

    def test_stabbing_finds_exactly_the_meeting_extents_by_list_position(self):
        rng = random.Random(2)
        for _ in range(3000):
            extents = random_extents(rng)
            stab = Stabbing(extents)
            for _ in range(10):
                lo = rng.randint(-4, 46)
                hi = lo + rng.randint(-2, 12)
                want = sorted(j for j, (s, e) in enumerate(extents) if s < hi and e > lo)
                self.assertEqual(sorted(stab.meeting(lo, hi)), want)
                self.assertEqual(sorted(stab.containing(lo)),
                                 sorted(j for j, (s, e) in enumerate(extents) if s < lo < e))

    def test_snap_matches_a_pass_over_every_span_in_order(self):
        def snap_by_scan(marks, spans):
            for mark in marks:
                for s, e in spans:
                    if s < mark["start"] < e:
                        mark["start"] = e
                    if s < mark["end"] < e:
                        mark["end"] = s
            return [m for m in marks if m["end"] > m["start"]]

        rng = random.Random(3)
        for _ in range(3000):
            spans = random_extents(rng)
            marks = [{"start": a, "end": a + rng.randint(-3, 15)} for a in (rng.randint(-1, 42) for _ in range(12))]
            by_scan, by_index = [dict(m) for m in marks], [dict(m) for m in marks]
            kept_scan = snap_by_scan(by_scan, list(spans))
            kept_index = wellformed._snap_marks_to_math_atoms(by_index, list(spans))
            self.assertEqual((by_scan, kept_scan), (by_index, kept_index))


if __name__ == "__main__":
    unittest.main()

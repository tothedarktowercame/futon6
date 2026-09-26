from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "efe_carpet_controls.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("efe_carpet_controls", SCRIPT)
assert SPEC and SPEC.loader
ctl = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ctl
SPEC.loader.exec_module(ctl)

NB, STEP = 7, 10
# 5x5 grid (step 10): zero everywhere except a max cell at (40, 40) and a
# mid cell at (20, 20) holding exactly the band-3 boundary value (3/7 of max).
GRID = [[0.0] * 5 for _ in range(5)]
GRID[4][4] = 100.0
GRID[2][2] = 100.0 * 3 / NB


class HubBandTest(unittest.TestCase):
    def test_max_point_is_top_band(self) -> None:
        self.assertEqual(ctl.hub_band(GRID, 100.0, NB, STEP, 40, 40), NB - 1)

    def test_zero_point_is_band_zero(self) -> None:
        self.assertEqual(ctl.hub_band(GRID, 100.0, NB, STEP, 0, 0), 0)

    def test_boundary_value(self) -> None:
        # value == 3/NB of fmax ⇒ int(3.0) == band 3 (boundary floors upward edge)
        self.assertEqual(ctl.hub_band(GRID, 100.0, NB, STEP, 20, 20), 3)
        # just under the boundary stays in band 2
        grid = [row[:] for row in GRID]
        grid[1][1] = 100.0 * 3 / NB - 1e-9
        self.assertEqual(ctl.hub_band(grid, 100.0, NB, STEP, 10, 10), 2)

    def test_clamps_to_grid(self) -> None:
        self.assertEqual(ctl.hub_band(GRID, 100.0, NB, STEP, 10**6, -10**6), 0)


class StatusClassTest(unittest.TestCase):
    def test_done_words(self) -> None:
        self.assertEqual(ctl.status_class("archived"), "done")
        self.assertEqual(ctl.status_class("COMPLETE (SUPERSEDED by M-other, 2026-03)"), "done")
        self.assertEqual(ctl.status_class("CLOSED 2026-06-12 (Joe's call)"), "done")
        self.assertEqual(ctl.status_class("retired."), "done")

    def test_open(self) -> None:
        self.assertEqual(ctl.status_class("IDENTIFY (2026-04-15): still mapping"), "open")

    def test_incomplete_is_open_not_done(self) -> None:
        # word-boundary matching: "incomplete" must NOT trip the "complete" word
        self.assertEqual(ctl.status_class("DERIVE incomplete; slice-2 still open"), "open")

    def test_finished_phase_of_open_mission_is_open(self) -> None:
        # live status lines from mission-activity.json, 2026-09-26
        self.assertEqual(ctl.status_class("HEAD complete; IDENTIFY draft pending operator acceptance"), "open")
        self.assertEqual(ctl.status_class("**INSTANTIATE (Stage 1 production-wired; simulation spike complete)**"), "open")
        self.assertEqual(ctl.status_class("ALL PHASES THROUGH DOCUMENT complete 2026-07-03"), "done")
        self.assertEqual(ctl.status_class("**Archived** 2026-06-01"), "done")

    def test_unknown(self) -> None:
        self.assertEqual(ctl.status_class(None), "unknown")
        self.assertEqual(ctl.status_class(""), "unknown")
        self.assertEqual(ctl.status_class("   "), "unknown")


if __name__ == "__main__":
    unittest.main()

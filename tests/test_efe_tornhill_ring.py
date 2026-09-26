from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "efe_tornhill_ring.py"
sys.path.insert(0, str(SCRIPT.parent))  # for futon6_config
SPEC = importlib.util.spec_from_file_location("efe_tornhill_ring", SCRIPT)
assert SPEC and SPEC.loader
ring = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ring
SPEC.loader.exec_module(ring)


def _fixture_report() -> dict:
    return {
        "generated": "2026-09-26T00:00:00+00:00",
        "window_days": 90,
        "repos": {
            "futon1": {"files": [
                {"path": "src/alpha/core.clj", "revs": 7, "hotspot": 100,
                 "born_in_window": False, "trend_ratio": 1.5},
                {"path": "src/beta/util.clj", "revs": 3, "hotspot": 40,
                 "born_in_window": True},
            ]},
        },
    }


MEASURED_ROW = {"mission": "M-alpha",
                "code": {"files": [["futon1", "src/alpha/core.clj"],
                                   ["futon1", "src/beta/util.clj"],
                                   ["futon9", "src/gone.clj"]]}}
UNMEASURED_ROW = {"mission": "M-gamma",
                  "code": {"files": [["futon9", "src/gone.clj"]]}}
NOLINK_ROW = {"mission": "M-delta", "code": {"files": []}}
NOLINK_ROW2 = {"mission": "M-epsilon"}


class MissionRingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.idx = ring.index(_fixture_report())

    def test_no_report(self) -> None:
        self.assertEqual(ring.mission_ring(MEASURED_ROW, None)["state"], "no-report")

    def test_no_link(self) -> None:
        self.assertEqual(ring.mission_ring(NOLINK_ROW, self.idx)["state"], "no-link")
        self.assertEqual(ring.mission_ring(NOLINK_ROW2, self.idx)["state"], "no-link")

    def test_unmeasured(self) -> None:
        r = ring.mission_ring(UNMEASURED_ROW, self.idx)
        self.assertEqual(r["state"], "unmeasured")
        self.assertEqual(r["n_files"], 1)

    def test_measured(self) -> None:
        r = ring.mission_ring(MEASURED_ROW, self.idx)
        self.assertEqual(r["state"], "measured")
        self.assertEqual(r["revs"], 10)
        self.assertEqual(r["hotspot"], 140)
        self.assertEqual(r["n_files_in_report"], 2)
        self.assertEqual(r["n_files"], 3)
        self.assertIsNone(r["seats"])  # no chat given
        trends = {t["path"]: t["trend"] for t in r["top"]}
        self.assertEqual(trends["futon1/src/beta/util.clj"], "new")
        self.assertEqual(trends["futon1/src/alpha/core.clj"], 1.5)

    def test_measured_seats_with_chat(self) -> None:
        chat = {"files": [{"repo": "futon1", "path": "src/alpha/core.clj",
                           "seats": {"claude-5": 2, "kimi-3": 1}}]}
        idx = ring.index(_fixture_report(), chat)
        r = ring.mission_ring(MEASURED_ROW, idx)
        self.assertEqual(r["seats"], 2)

    def test_ring_svg_measured_names_report(self) -> None:
        r = ring.mission_ring(MEASURED_ROW, self.idx)
        meta = {"filename": "tornhill-2026-09-26.json",
                "generated": "2026-09-26T00:00:00+00:00", "window_days": 90}
        svg = ring.ring_svg(10, 20, 5.0, "M-alpha", r, meta, 140)
        self.assertIn("tornhill-2026-09-26.json", svg)
        self.assertIn("revs 10", svg)
        self.assertIn("new", svg)
        self.assertIn("2 of 3 files", svg)

    def test_ring_svg_no_report_is_empty(self) -> None:
        svg = ring.ring_svg(0, 0, 5.0, "M-x", {"state": "no-report"}, None, 1)
        self.assertEqual(svg, "")

    def test_load_report_absent(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ring.load_report(d))

    def test_load_report_picks_newest(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for name, gen in [("tornhill-2026-09-01.json", "old"),
                              ("tornhill-2026-09-26.json", "new")]:
                rep = _fixture_report()
                rep["generated"] = gen
                Path(d, name).write_text(json.dumps(rep))
            self.assertEqual(ring.load_report(d)["generated"], "new")


if __name__ == "__main__":
    unittest.main()

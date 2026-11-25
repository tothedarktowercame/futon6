"""Fixtures-only tests for archipelago_illumination: rates, gaps, styles,
scale, constructed filter. No HTML, no live data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import archipelago_illumination as ai


def test_rate_normalisation_beats_raw_ranking():
    # 10 exposures in a 100-turn window vs 100 in a 10,000-turn window:
    # rates 100 vs 10 per 1,000 turns; raw ranks them the other way.
    exposure = {"P": {"2026-06-01": 10}, "Q": {"2026-06-01": 100}}
    turns = {"2026-06-01": 100}
    rates, _ = ai.window_rates(exposure, turns, [], ["2026-06-01"],
                               window_days=1)
    assert rates["2026-06-01"]["P"] == (100.0, 10)
    turns2 = {"2026-06-01": 10000}
    rates2, _ = ai.window_rates({"Q": {"2026-06-01": 100}}, turns2, [],
                                ["2026-06-01"], window_days=1)
    assert rates2["2026-06-01"]["Q"] == (10.0, 100)


def test_gap_day_is_unknown_not_zero_turns():
    # window of 2 days: one known (100 turns, 10 exposures), one gap.
    exposure = {"P": {"2026-06-01": 10}}
    turns = {"2026-06-01": 100, "2026-06-02": 5000}
    rates, gaps = ai.window_rates(exposure, turns, ["2026-06-02"],
                                  ["2026-06-02"], window_days=2)
    assert gaps["2026-06-02"] == 1
    # rate over the known day only: 100, not 1000*10/5100 = 1.96
    assert rates["2026-06-02"]["P"][0] == 100.0


def test_unknown_pattern_vs_known_zero_are_distinguishable():
    o0, dash0, unk0 = ai.road_style("absent", {}, 100.0, {"present"})
    o1, dash1, unk1 = ai.road_style("present", {}, 100.0, {"present"})
    assert unk0 and dash0 and o0 == ai.OPACITY_MIN
    assert not unk1 and not dash1 and o1 == ai.OPACITY_MIN
    assert (dash0, unk0) != (dash1, unk1)


def test_scale_is_absolute_across_windows():
    rates = {"w1": {"P": (10.0, 5)}, "w2": {"P": (900.0, 400),
                                            "Q": (1000.0, 500)}}
    ceil = ai.scale_max(rates, "rate", percentile=100)
    assert ceil == 1000.0
    # w1 uses the same ceiling: P's opacity in w1 is well below its
    # per-window-max value (no per-window renormalisation).
    assert ai.opacity(10.0, ceil) < ai.opacity(10.0, 10.0)


def test_road_constructed_after_window_end_is_excluded():
    assert not ai.road_drawable("2026-09-15T10:00:00+00:00", "2026-09-01")
    assert ai.road_drawable("2026-08-15T10:00:00+00:00", "2026-09-01")
    assert ai.road_drawable(None, "2026-09-01")  # unknown: drawn, not excluded


def test_window_ends_are_mondays_within_range():
    ends = ai.window_ends("2026-04-13", "2026-06-02")
    from datetime import date
    assert all(date.fromisoformat(e).weekday() == 0 for e in ends)
    assert ends[0] >= "2026-05-13" and ends[-1] <= "2026-06-02"


def test_strand_offsets_centred():
    rows = [{"a": "M-a", "b": "M-b", "pattern": p} for p in ("P", "Q", "R")]
    offs = ai.strand_offsets(rows)
    assert sorted(offs.values()) == [-2.0, 0.0, 2.0]


def test_page_dashes_only_unknown_patterns():
    # The page applies its own style rule in JS; road_style() above is the
    # spec. Kimi's first page dashed known-zero roads too (unknown || !rec),
    # merging the two states this prototype must keep apart.
    assert "(!focused && unknown) ? '4,3'" in ai.SCRIPT
    assert "unknown || !rec" not in ai.SCRIPT

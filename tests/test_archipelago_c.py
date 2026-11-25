"""Fixtures-only tests for archipelago_c: classification, aggregation,
since-c rates, islets, and a page-rule guard. No live data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import archipelago_c as ac
import archipelago_tide as tide


def _sub(closed):
    return lambda m, c: m in closed


def _rows():
    return [
        dict(a="M-A", b="M-B", pattern="P", _ia=0, _ib=0, _a_exposed=True),
        dict(a="M-A", b="M-C", pattern="P", _ia=0, _ib=1, _a_exposed=True),
        dict(a="M-C", b="M-D", pattern="Q", _ia=1, _ib=None, _a_exposed=True),
        dict(a="M-D", b="M-D2", pattern="R", _ia=None, _ib=None, _a_exposed=False),
    ]


def test_classification_four_classes():
    island_of = {"M-A": 0, "M-B": 0, "M-C": 1}
    cls = ac.classify_roads(_rows(), island_of, _sub({"M-D", "M-D2"}), "2026-09-01")
    assert [r["b"] for r in cls["interior"]] == ["M-B"]
    assert [r["b"] for r in cls["ferry"]] == ["M-C"]
    assert [r["b"] for r in cls["tether"]] == ["M-D"]
    assert [r["b"] for r in cls["sunken"]] == ["M-D2"]


def test_ferry_aggregation_counts_distinct_patterns():
    rows = [
        dict(a="M-A", b="M-C", pattern="P", _ia=0, _ib=1),
        dict(a="M-B", b="M-C", pattern="Q", _ia=0, _ib=1),
        # same pattern again on a different mission pair: k counts it once
        dict(a="M-A", b="M-E", pattern="P", _ia=0, _ib=1),
    ]
    ferries = ac.aggregate_ferries(rows)
    assert len(ferries) == 1
    f = ferries[0]
    assert len(f["patterns"]) == 2  # distinct patterns, not 3 rows
    assert len(f["pairs"]) == 3     # mission pairs are all kept


def test_rates_since_excludes_exposure_before_c():
    exposure = {"P": {"2026-04-13": 50, "2026-09-10": 10}}
    turns = {"2026-04-13": 100, "2026-09-10": 100}
    rates, starts, _ = ac.rates_since(exposure, turns, [],
                                      ["2026-09-01"], "2026-04-13", "2026-09-30")
    # since 2026-09-01 only the September exposure counts: 100/1k, not 600
    assert rates["2026-09-01"]["P"] == (100.0, 10)
    assert starts["2026-09-01"] == "2026-09-01"
    # a cutoff before the exposure table starts is clamped to first_day
    rates2, starts2, _ = ac.rates_since(exposure, turns, [],
                                        ["2026-01-01"], "2026-04-13", "2026-09-30")
    assert starts2["2026-01-01"] == "2026-04-13"
    assert rates2["2026-01-01"]["P"] == (300.0, 60)


def test_rates_since_gap_days_are_unknown_not_zero():
    exposure = {"P": {"2026-06-01": 10}}
    turns = {"2026-06-01": 100, "2026-06-02": 99999}
    rates, _, gaps = ac.rates_since(exposure, turns, ["2026-06-02"],
                                    ["2026-06-01"], "2026-06-01", "2026-06-30")
    assert rates["2026-06-01"]["P"][0] == 100.0  # not 10*1000/100099
    assert gaps["2026-06-01"] == 1


def test_exposed_mission_on_water_gets_islet_id():
    land = {(0, 0), (1, 0)}  # one component; M-A's hub inside, M-B's outside
    hub_cells = {"M-A": (0, 0), "M-B": (50, 50)}
    island_of, comps, n_islets = ac.island_membership(
        land, hub_cells, _sub(set()), "2026-09-01")
    assert island_of["M-A"] == 0
    assert island_of["M-B"] == "islet:M-B"
    assert n_islets == 1
    # islet is its own island: A and B are not in the same component
    assert island_of["M-A"] != island_of["M-B"]


def test_page_rule_guard_interior_respects_focus():
    # The page re-implements class overrides in JS; guard the rule that
    # interior strands are only drawn when nothing is focused (focus mode
    # draws its own strand set), and tethers stay off in focus mode.
    assert "showInterior && focus === null" in ac.SCRIPT
    assert "gFerry.style.display = focus !== null ? 'none' : ''" in ac.SCRIPT


def test_page_dims_whole_missions_via_the_shared_block():
    # C first copied A's pre-fix dimming (circle[data-m]: one scope dot per
    # mission). Both pages must use the shared block from archipelago_tide.
    assert "__HUB_DIM_JS__" in ac.SCRIPT
    assert "circle[data-m]" not in ac.SCRIPT
    import archipelago_tide as at
    assert "querySelectorAll('[data-m]')" in at.HUB_DIM_JS

"""Fixtures-only tests for archipelago_tide: field / threshold / components.
No HTML, no live data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import archipelago_tide as at

STEP, SIGMA = 40, 70.0


def _cluster(cx, cy, n=6):
    """n unit-weight points tightly around (cx, cy)."""
    return [(cx + (i % 3) * 8 - 8, cy + (i // 3) * 8 - 8, 1.0)
            for i in range(n)]


def _field(points, gw=20, gh=20):
    f = [[0.0] * gw for _ in range(gh)]
    for x, y, w in points:
        at.gaussian_scatter(f, x, y, w, step=STEP, sigma=SIGMA)
    return f


def _clusters_fields():
    # cluster P at grid (5,5), cluster Q at grid (14,14) — far apart
    p = _cluster(5 * STEP, 5 * STEP)
    q = _cluster(14 * STEP, 14 * STEP)
    return p, q


def test_tide_empties_old_clusters_not_terrain():
    p, q = _clusters_fields()
    theta = 0.2
    # cutoff 2026-02: both clusters counted -> 2 islands
    both = _field(p + q)
    # cutoff 2026-06: P's March scopes expired, only Q counted -> 1 island
    only_q = _field(q)
    # cutoff 2026-10: nothing counted -> 0 islands
    none = _field([])
    for f, want in ((both, 2), (only_q, 1), (none, 0)):
        comps = at.components(at.land_cells(f, theta))
        assert len(comps) == want
    if at.components(at.land_cells(only_q, theta)):
        (comp,) = at.components(at.land_cells(only_q, theta))
        # the surviving island contains Q's cell, not P's
        assert (14, 14) in comp and (5, 5) not in comp


def test_undated_scope_stays_land_at_every_cutoff():
    pts = _cluster(10 * STEP, 10 * STEP, n=4)  # undated: never expire
    f = _field(pts)
    land = at.land_cells(f, 0.2)
    assert (10, 10) in land
    # and an empty tide (all dated scopes expired) keeps it: the field
    # above IS the undated-only field
    assert at.components(land)


def test_theta_is_one_constant_for_all_cutoffs():
    p, q = _clusters_fields()
    low = _field(p + q)
    baseline = at.land_cells(low, 0.2)
    theta, j = at.choose_theta(low, baseline)
    assert j == 1.0
    # the SAME theta is then applied at every cutoff (no per-cutoff
    # renormalisation): fields are never rescaled by their own max.
    high = _field(q)
    assert at.land_cells(high, theta)  # Q still land
    fmax = max(max(r) for r in high)
    rescaled = [[v / fmax * theta for v in row] for row in high]
    # the bad case: rescaling so the map never empties — a lone faint
    # scope would stay "land" forever. Assert we do NOT do that.
    faint = _field([(14 * STEP, 14 * STEP, 0.05)])
    assert at.land_cells(faint, theta) == set()


def test_weekly_cutoffs_are_mondays():
    c = at.weekly_cutoffs("2026-03-04", "2026-03-20")
    assert c == ["2026-03-02", "2026-03-09", "2026-03-16"]
    from datetime import date
    for d in c:
        assert date.fromisoformat(d).weekday() == 0


def test_components_four_connected():
    cells = {(0, 0), (1, 0), (1, 1), (5, 5), (0, 3)}
    comps = sorted((sorted(c) for c in at.components(cells)))
    assert len(comps) == 3
    assert {(0, 0), (1, 0), (1, 1)} in [set(c) for c in comps]


def _dated(points, day):
    return [{"x": x, "y": y, "dated": day, "mission": "M"} for x, y, _ in points]


def test_tide_comes_from_dates_through_the_real_field():
    # The date filtering itself, not hand-built fields: P worked in March,
    # Q in September. Bad case: a field that ignores dates (or uses scope
    # count) keeps 2 islands at every cutoff.
    p, q = _clusters_fields()
    pts = _dated(p, "2026-03-10T12:00:00+00:00") + _dated(q, "2026-09-10T12:00:00+00:00")
    low = _field(p + q)
    low_land = at.land_cells(low, 0.2)
    got = {c: (len(at.components(at.tide_land(f, low_land, 0.2))), n)
           for c, f, n in at.fields_by_cutoff(
               pts, ["2026-02-02", "2026-06-01", "2026-10-05"], gw=20, gh=20)}
    assert got == {"2026-02-02": (2, 0), "2026-06-01": (1, 6),
                   "2026-10-05": (0, 12)}


def test_undated_scope_never_expires_in_the_real_field():
    pts = [{"x": 400, "y": 400, "dated": None, "mission": "M"}]
    low_land = at.land_cells(_field([(400, 400, 1.0)]), 0.5)
    for c, f, n in at.fields_by_cutoff(pts, ["2026-01-05", "2027-01-04"], gw=20, gh=20):
        assert n == 0 and at.tide_land(f, low_land, 0.5) == low_land


def test_hub_submersion_is_by_its_own_dates_not_its_size():
    # a mission with one recent scope is above water, however small;
    # an undated scope or no scopes at all never submerges it
    assert not at.mission_submerged(["2026-09-20T00:00:00+00:00"], "2026-09-14")
    assert at.mission_submerged(["2026-09-20T00:00:00+00:00"], "2026-09-21")
    assert not at.mission_submerged(["2026-01-01T00:00:00+00:00", None], "2026-11-23")
    assert not at.mission_submerged([], "2026-11-23")

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

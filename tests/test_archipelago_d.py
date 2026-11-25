"""Fixtures-only tests for archipelago_d: island labels, zoom thresholds,
page-rule guard over the preset selectors."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import archipelago_d as ad


def test_island_label_is_exposed_mission_with_most_scopes():
    counts = {"M-alpha": 30, "M-beta": 12, "M-gamma": 41}
    assert ad.choose_label(["M-alpha", "M-beta"], counts) == "M-alpha"
    assert ad.choose_label(["M-alpha", "M-gamma"], counts) == "M-gamma"
    # a submerged mission is never in `members` (island_membership in
    # archipelago_c filters them out, tested there), so it cannot label:
    # simulate by exclusion — the label of the remaining members is beta.
    assert ad.choose_label(["M-beta"], counts) == "M-beta"


def test_zoom_level_thresholds():
    assert ad.zoom_level(2500) == "far"
    assert ad.zoom_level(2001) == "far"
    assert ad.zoom_level(2000) == "mid"   # boundary: far is W > 2000
    assert ad.zoom_level(1200) == "mid"
    assert ad.zoom_level(801) == "mid"
    assert ad.zoom_level(800) == "near"   # boundary: near is W <= 800
    assert ad.zoom_level(600) == "near"


def test_contrast_ratio_sanity():
    assert ad.contrast_ratio("#000000", "#ffffff") == 21.0
    assert ad.contrast_ratio(ad.PAPER, ad.PAPER) == 1.0
    assert ad.contrast_ratio("#33475c", "#efe9dc") > 3.0


def test_page_rule_guard_selectors_resolve_in_baseline():
    html = ad.BASELINE_HTML.read_text()
    missing = []
    for sel in ad.SELECTORS_USED:
        if sel.startswith("#"):
            if f'id="{sel[1:]}"' not in html and f'"{sel[1:]}"' not in html:
                missing.append(sel)
        elif sel.startswith("."):
            if sel[1:] not in html:
                missing.append(sel)
        elif sel.startswith("line[stroke="):
            color = sel.split('"')[1]
            if f'stroke="{color}"' not in html:
                missing.append(sel)
        elif sel.startswith("circle[stroke="):
            color = sel.split('"')[1]
            if f'stroke="{color}"' not in html:
                missing.append(sel)
        elif sel.startswith("text[fill="):
            color = sel.split('"')[1]
            if f'text' not in html or f'fill="{color}"' not in html:
                missing.append(sel)
    # c-* layers are added by archipelago_c's compose (not the baseline);
    # d-terrain-fill is tagged by D's own compose_page splice. Check both
    # against their sources.
    import inspect
    import archipelago_c as ac
    c_src = inspect.getsource(ac.compose_page)
    c_missing = [s for s in missing if s.startswith("#c-")
                 and f'id="{s[1:]}"' not in c_src]
    d_src = inspect.getsource(ad.compose_page)
    d_ok = '<g id="d-terrain-fill">' in d_src
    unexpected = [s for s in missing
                  if not s.startswith("#c-") and s != "#d-terrain-fill"
                  and not s.startswith(".d-")]
    assert not unexpected, unexpected
    assert not c_missing, c_missing
    assert d_ok, "compose_page no longer tags the terrain fill group"
    for s in (".d-hub-label", ".d-island-label"):
        assert s[1:] in ad.SCRIPT, f"{s} not created by D's script"
    # and every selector in the CSS is in the declared list
    css_sels = set(re.findall(r'(#(?:[a-z0-9-]+)|\.(?:[a-z0-9-]+)'
                              r'|line\[stroke="[^"]+"\]'
                              r'|circle\[stroke="[^"]+"\]'
                              r'|text\[fill="[^"]+"\])', ad.STYLE_CSS))
    css_sels = {s for s in css_sels
                if not re.fullmatch(r'#[0-9a-fA-F]{3,6}', s)  # hex colours
                and not re.fullmatch(r'\.[0-9].*', s)}        # decimals
    declared = set(ad.SELECTORS_USED)
    undeclared = {s for s in css_sels
                  if not any(s == d or (s[0] in "#." and s == d) for d in declared)}
    assert not undeclared, undeclared


def test_page_zoom_thresholds_come_from_python():
    # The page's level() must use ZOOM_FAR / ZOOM_MID, not its own copy.
    assert "const FAR = __ZOOM_FAR__, MID = __ZOOM_MID__;" in ad.SCRIPT

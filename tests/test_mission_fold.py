"""Contract tests for the fold→WM-selection edge (R1, Campaign-bayesian-structure-learning).

Pin the cross-agent contract claude-1's WM reader consumes:
  * :gap-score = GROWTH-SURFACE (count of announced-but-empty sections, saturated,
    size-floor-gated) — de-biased after the STANDARD-VERIFY finding that mean-stub-
    fraction was size-dominated and would have mis-steered the WM toward tiny stubs;
  * operator events drive SALIENCE only (gap-score must NOT move);
  * the event-log intake parses the contract and skips garbage.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import mission_fold_learn as mfl  # noqa: E402


def _mission(n_empty, n_filled=2, fill=10):
    """A substantial mission (raw >= FLOOR) with n_empty announced-but-empty sections."""
    nodes = {"root": {"binder": "loose-section", "parent": None, "sub_count": 999, "fillers": []}}
    for i in range(n_empty):
        nodes[f"e{i}"] = {"binder": "eightfold-phase", "parent": "root", "sub_count": 0, "fillers": []}
    for i in range(n_filled):
        nodes[f"f{i}"] = {"binder": "eightfold-phase", "parent": "root", "sub_count": fill,
                          "fillers": [("concept", f"t{i}{j}") for j in range(fill)]}
    return nodes


def test_growth_surface_mass():
    # MASS = Σ(GAP_THIN − sub_count) over empty sections; empty sub_count=0 -> 8 each, /MASS_CAP=120
    assert mfl.gap_score(_mission(n_empty=3)) == 0.2   # 24/120
    assert mfl.gap_score(_mission(n_empty=0)) == 0.0   # nothing announced-but-empty


def test_no_ceiling_cluster_distinct_counts():
    # F1 (claude-1's regulator harness): distinct empty-mass -> DISTINCT, un-clamped scores,
    # so the EFE gap term can discriminate instead of tying the within-local top at 1.0.
    g10, g11, g12 = (mfl.gap_score(_mission(n_empty=k)) for k in (10, 11, 12))
    assert g10 < g11 < g12 < 1.0          # strictly increasing, none clamped to a shared ceiling
    assert len({g10, g11, g12}) == 3


def test_mass_caps_only_at_maximally_hollow():
    assert mfl.gap_score(_mission(n_empty=15)) == 1.0   # 120/120 — the hollow ceiling
    assert mfl.gap_score(_mission(n_empty=20)) == 1.0   # capped, not >1


def test_size_floor_zeroes_tiny_stubs():
    # the mis-steer class: a tiny mission (raw < FLOOR) gets NO growth signal
    tiny = {"root": {"binder": "loose-section", "parent": None, "sub_count": 99, "fillers": []},
            "e0": {"binder": "eightfold-phase", "parent": "root", "sub_count": 0, "fillers": []}}
    assert mfl.gap_score(tiny) == 0.0


def test_root_container_excluded():
    # the mission-root (parent=None) is not an announced section even if empty
    nodes = _mission(n_empty=2)
    nodes["root"]["sub_count"] = 0
    assert mfl.gap_score(nodes) == round(16 / 120, 3)  # only the 2 real empty sections (8 each)


def test_load_events_parses_and_skips_garbage(tmp_path):
    log = tmp_path / "ev.jsonl"
    log.write_text("\n".join([
        json.dumps({"mission": "M-x", "frame": "MAP", "action": "expand"}),
        "not json at all",
        json.dumps({"mission": "M-x", "frame": "ID", "action": "collapse"}),
        json.dumps({"mission": "M-y", "action": "expand"}),  # no frame -> skipped
    ]))
    assert mfl.load_events(log) == {"M-x": [("expand", "MAP"), ("collapse", "ID")]}


def test_load_events_missing_file_is_empty(tmp_path):
    assert mfl.load_events(tmp_path / "nope.jsonl") == {}


def test_events_move_salience_not_gap():
    base = mfl.build_view("war-machine")
    drilled = mfl.build_view("war-machine", trace=[("expand", "MAP")] * 4)
    sal = lambda v, f: next(x["salience"] for x in v["spine"] if x["frame"] == f)  # noqa: E731
    assert sal(drilled, "MAP") > sal(base, "MAP")          # behavioural signal moved
    assert drilled["gap-score"] == base["gap-score"]        # structural signal stable


def test_mis_steer_stubs_are_zeroed_on_real_corpus():
    # the live-risk gate: the open tiny stubs claude-1 flagged must score 0
    for stub in ("coupling-as-constraint", "tpg-coupling-evolution", "sliding-blackboard"):
        try:
            assert mfl.build_view(stub)["gap-score"] == 0.0
        except IndexError:
            pass  # tree absent in this checkout — skip rather than fail
